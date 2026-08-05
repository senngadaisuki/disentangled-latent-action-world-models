import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import wandb
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from utils.load_mixed_datasets import (
    create_dataset_configs_from_config,
)
from utils.mixed_dataset import create_mixed_dataloader
from torch.utils.data import DataLoader, IterableDataset
from einops import rearrange
import os.path as osp
from pathlib import Path
from datetime import timedelta
from accelerate import Accelerator
from accelerate.utils import DistributedType
from accelerate.logging import get_logger
from accelerate import InitProcessGroupKwargs, DistributedDataParallelKwargs
from models.RAE import RAE
from omegaconf import DictConfig, OmegaConf
import hydra
from utils.utils import default
from typing import Optional
import torchvision

os.environ["WANDB_START_METHOD"] = "thread"
logger = get_logger(__name__)


class _DMCObsWrapper(IterableDataset):
    """
    Module-level wrapper for DMC ChainDataset.
    Renames 'images' → 'observations' to match the training loop's expected key.
    Must be at module level (not inside a method) so it can be pickled when
    DataLoader uses multiprocessing_context='spawn'.
    """
    def __init__(self, ds):
        self._ds = ds

    def __iter__(self):
        for sample in self._ds:
            yield {"observations": sample["images"]}


def cycle(dl: DataLoader, skipped_dl: Optional[DataLoader] = None):
    if skipped_dl is not None:
        for data in skipped_dl:
            yield data
    while True:
        for data in dl:
            yield data

class Trainer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
            
        process_group_kwargs = InitProcessGroupKwargs(
            timeout=timedelta(seconds=self.cfg.accelerator.timeout_seconds),
        )
        dist_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            log_with="wandb", kwargs_handlers=[process_group_kwargs, dist_kwargs]
        )
        logger.info(f"Mixed precision: {self.accelerator.mixed_precision}")
        
        # Use cfg to initialize parameters
        self.exp_name = self.cfg.exp_name
        self.phase = self.cfg.phase
        self.grad_accum_every = self.cfg.grad_accum_every
        self.max_grad_norm = self.cfg.max_grad_norm
        self.save_model_every = self.cfg.save_model_every
        self.save_milestone_every = self.cfg.save_milestone_every
        self.milestone_optim_state = self.cfg.milestone_optim_state
        self.val_every_n_steps = self.cfg.val_every_n_steps
        self.num_val_batches_to_log = self.cfg.num_val_batches_to_log
        self.num_val_samples_to_save = min(self.cfg.num_val_samples_to_save, default(self.cfg.val_batch_size, self.cfg.batch_size))

        self.num_train_steps = self.cfg.num_train_steps
        self.current_step = 0
        self.current_val_step = 0

        self.lr_dict = {
            "inverse_lr": self.cfg.optimizer.inverse_lr,
            "world_lr": self.cfg.optimizer.world_lr,
        }
        self._set_seed_everywhere(self.cfg.seed)

        self.work_dir = Path(hydra.utils.to_absolute_path(self.cfg.work_dir))

        # all processes use the work_dir from the main process
        if torch.distributed.is_initialized():
            objs = [str(self.work_dir)]
            torch.distributed.broadcast_object_list(objs, 0)
            self.work_dir = Path(objs[0])
        self.accelerator.wait_for_everyone()
        logger.info("Saving to {}".format(self.work_dir))

        self.wandb_name = self.cfg.wandb.name + "-" + self.cfg.exp_name
        self.wandb_enabled = self.cfg.wandb.enabled

        md_cfg = self.cfg.data.get("multidataset", None)
        use_dmc = (
            md_cfg is not None
            and getattr(md_cfg, "use_dmc_streaming", False)
        )
        if use_dmc:
            self._setup_dmc_streaming_loaders()
        elif self.cfg.data.use_multidataset:
            self._setup_multidataset_loaders()
        else:
            self._setup_loaders()
        self._init_tracker()
        self._init_model(self.phase)

    def _init_tracker(self):
        self.accelerator.init_trackers(
            project_name=self.cfg.wandb.project,
            init_kwargs={
                "wandb": {
                    "reinit": False,
                    "settings": {"start_method": "thread"},
                    "name": self.wandb_name,
                    "mode": "online" if self.wandb_enabled else "disabled",
                    "save_code": True,
                    "config": OmegaConf.to_container(self.cfg, resolve=True, throw_on_missing=True)
                },
            },
        )
        if self.accelerator.is_main_process:
            self.wandb_run = self.accelerator.get_tracker("wandb", unwrap=True)
            logger.info("wandb run url: %s", self.wandb_run.get_url())

    def _set_seed_everywhere(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    def _init_model(self, phase):
        self.rae = RAE(
            encoder_cls=self.cfg.RAE.encoder_cls,
            encoder_config_path=self.cfg.RAE.encoder_config_path,
            encoder_input_size=self.cfg.RAE.encoder_input_size,
            encoder_params=self.cfg.RAE.encoder_params,
            decoder_config_path=self.cfg.RAE.decoder_config_path,
            decoder_patch_size=self.cfg.RAE.decoder_patch_size,
            pretrained_decoder_path=self.cfg.RAE.pretrained_decoder_path,  
            reshape_to_2d=self.cfg.RAE.reshape_to_2d,
            noise_tau=self.cfg.RAE.noise_tau,
            normalization_stat_path=self.cfg.RAE.normalization_stat_path,  
        )
        self.rae.eval()
        self.rae.to(self.device)
        for p in self.rae.parameters(): p.requires_grad_(False)
        self.rae = self.accelerator.prepare(self.rae)
        logger.info(f'RAE initialized.')

        # Phase 1 Training
        if phase == 1:
            self.structure_encoder = hydra.utils.instantiate(self.cfg.structure_encoder).to(self.device)
            self.content_fusion = hydra.utils.instantiate(self.cfg.content_fusion).to(self.device)
            self.model = hydra.utils.instantiate(self.cfg.world_model,
                                                 structure_encoder=self.structure_encoder,
                                                 content_fusion=self.content_fusion,
                                                 phase=phase).to(self.device)

            self.model.apply(self.model._init_weights)

            self.optimizer, self.scheduler = \
                self.model.configure_optimizers(weight_decay=self.cfg.optimizer.weight_decay, lr=self.lr_dict, betas=tuple(self.cfg.optimizer.betas), T_max=self.cfg.optimizer.T_max)
            
            self.model, self.optimizer, self.scheduler = \
                    self.accelerator.prepare(self.model, self.optimizer, self.scheduler)
        # Phase 2 Training
        elif phase == 2:
            self.structure_encoder = hydra.utils.instantiate(self.cfg.structure_encoder).to(self.device)
            self.content_fusion = hydra.utils.instantiate(self.cfg.content_fusion).to(self.device)
            self.model = hydra.utils.instantiate(self.cfg.world_model,
                                                 structure_encoder=self.structure_encoder,
                                                 content_fusion=self.content_fusion,
                                                 phase=phase).to(self.device)
            self.model.apply(self.model._init_weights)

            model_ckpt = hydra.utils.to_absolute_path(self.cfg.model.phase1_ckpt)
            data = torch.load(model_ckpt, map_location="cpu")

            # Handle potential mismatch if model was saved directly vs. via get_state_dict
            if "model" in data:
                model_state = data["model"]
            elif "module" in data:  # If it was a DDP model state_dict
                model_state = data["module"]
            else:
                model_state = data  # Assume entire data is model state if 'model' key is missing

            self.model.load_state_dict(model_state, strict=True)
            logger.info(f'load pre-trained weights finished.')

            self.optimizer, self.scheduler = \
                self.model.configure_optimizers(weight_decay=self.cfg.optimizer.weight_decay, lr=self.lr_dict, betas=tuple(self.cfg.optimizer.betas), T_max=self.cfg.optimizer.T_max)
            
            self.model, self.optimizer, self.scheduler = \
                    self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

            # model_ckpt = hydra.utils.to_absolute_path(self.cfg.model.phase1_ckpt)
            # self.load(model_ckpt)

        elif phase == 3:
            self.structure_encoder = hydra.utils.instantiate(self.cfg.structure_encoder).to(self.device)
            self.content_fusion = hydra.utils.instantiate(self.cfg.content_fusion).to(self.device)

            action_decoder_ckpt = self.cfg.model.get("action_decoder_ckpt", None)
            if action_decoder_ckpt is None:
                raise ValueError(
                    "Phase 3 requires action-conditioned training. Please provide "
                    "`model.action_decoder_ckpt=/path/to/action_decoder.pt` and use a dataset "
                    "that returns an `action` field, such as `data=robotask`."
                )

            self.Action_decoder = hydra.utils.instantiate(self.cfg.decoder).to(self.device)

            self.model = hydra.utils.instantiate(self.cfg.world_model,
                                                 structure_encoder=self.structure_encoder,
                                                 content_fusion=self.content_fusion,
                                                 Action_decoder=self.Action_decoder,
                                                 phase=phase).to(self.device)
            self.model.apply(self.model._init_weights)

            action_decoder_ckpt = hydra.utils.to_absolute_path(action_decoder_ckpt)
            data = torch.load(action_decoder_ckpt, map_location="cpu")
            self.model.Action_decoder.load_state_dict(data['model_state_dict'], strict=True)
            logger.info(f'load action decoder weights finished.')
            
            model_ckpt = hydra.utils.to_absolute_path(self.cfg.model.phase2_ckpt)
            data = torch.load(model_ckpt, map_location="cpu")

            # Handle potential mismatch if model was saved directly vs. via get_state_dict
            if "model" in data:
                model_state = data["model"]
            elif "module" in data:  # If it was a DDP model state_dict
                model_state = data["module"]
            else:
                model_state = data  # Assume entire data is model state if 'model' key is missing

            self.model.load_state_dict(model_state, strict=False)
            logger.info(f'load pre-trained weights finished.')

            self.optimizer, self.scheduler = \
                self.model.configure_optimizers(weight_decay=self.cfg.optimizer.weight_decay, lr=self.lr_dict, betas=tuple(self.cfg.optimizer.betas), T_max=self.cfg.optimizer.T_max)
            
            self.model, self.optimizer, self.scheduler = \
                    self.accelerator.prepare(self.model, self.optimizer, self.scheduler)

        else:
            raise ValueError("Invalid phase. Choose from 1, 2, or 3.")

    def _setup_loaders(self):
        from utils.trajectorydataset import TrajectoryDataset
        root_dir = hydra.utils.to_absolute_path(self.cfg.data.root_dir)

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        train_dataset = TrajectoryDataset(
            root_dir=root_dir,
            mode='train',
            split_ratio=1,
            seed=self.cfg.seed,
            transform=transform,
            seq_len=self.cfg.data.seq_len,
            stride=self.cfg.data.seq_len,
        )

        self.train_loader = DataLoader(train_dataset, batch_size=self.cfg.data.batch_size, shuffle=self.cfg.data.shuffle_train, num_workers=self.cfg.data.num_workers, pin_memory=self.cfg.data.pin_memory)
        self.val_loader = DataLoader(train_dataset, batch_size=self.cfg.data.val_batch_size, shuffle=False, num_workers=self.cfg.data.num_workers, pin_memory=self.cfg.data.pin_memory)

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.val_loader = self.accelerator.prepare(self.val_loader)

        self.train_dl_iter = cycle(self.train_loader)
        self.val_dl_iter = cycle(self.val_loader)

    def _setup_dmc_streaming_loaders(self):
        """
        Build DataLoaders from one or more DMCVisionDataset (IterableDataset)
        instances chained together via torch.utils.data.ChainDataset.

        Each entry in cfg.data.multidataset.dmc_domains corresponds to one
        DMC domain (e.g. walker / cheetah / humanoid).  All domains share the
        same batch_size, seq_len and num_workers from the multidataset config.

        Output batch key is 'observations' so that the existing train_step /
        run_validation_and_log code works without any further changes.
        """
        from torch.utils.data import ChainDataset
        from utils.datasets_dmc import DMCVisionDataset

        md_cfg = self.cfg.data.multidataset

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        seq_len    = int(md_cfg.seq_len)
        batch_size = int(md_cfg.batch_size)
        val_batch  = int(md_cfg.val_batch_size)
        nworkers   = int(md_cfg.num_workers)
        pin_mem    = bool(md_cfg.pin_memory)
        shuffle    = bool(md_cfg.shuffle_train)

        train_datasets, val_datasets = [], []
        for dom in md_cfg.dmc_domains:
            common_kwargs = dict(
                data_dir            = dom.data_dir,
                domain_name         = dom.domain_name,
                policy_level        = dom.get("policy_level", "expert"),
                difficulty          = dom.get("difficulty", "none"),
                dynamic_distractors = dom.get("dynamic_distractors", False),
                seq_len             = seq_len,
                img_height          = dom.get("img_height", 64),
                img_width           = dom.get("img_width", 64),
                obs_vrange          = tuple(dom.get("obs_vrange", [0.0, 1.0])),
                transform           = transform,
                stride              = dom.get("stride", seq_len),
                shuffle_buffer_size = dom.get("shuffle_buffer_size", 500),
                max_samples         = dom.get("max_samples", None),
            )
            train_datasets.append(DMCVisionDataset(
                **common_kwargs, split=dom.get("train_split", "train"),
                shuffle=shuffle,
            ))
            val_datasets.append(DMCVisionDataset(
                **common_kwargs, split=dom.get("val_split", "train"),
                shuffle=False,
            ))
            logger.info(
                f"[DMC streaming] domain={dom.domain_name} "
                f"difficulty={dom.get('difficulty','none')} "
                f"policy={dom.get('policy_level','expert')}"
            )

        # Chain all domain datasets into a single iterable stream
        train_chain = ChainDataset(train_datasets)
        val_chain   = ChainDataset(val_datasets)

        # Scale down workers per GPU to avoid OOM: with N GPUs each spawning
        # nworkers TF processes the total memory cost grows fast.
        num_gpus = max(1, self.accelerator.num_processes)
        workers_per_gpu = max(2, nworkers // num_gpus)
        logger.info(
            f"[DMC streaming] num_workers per GPU: {workers_per_gpu} "
            f"(total_workers_requested={nworkers}, num_gpus={num_gpus})"
        )

        # Use 'spawn' instead of the default 'fork' start method.
        # TensorFlow is NOT fork-safe: forking after CUDA is initialised causes
        # TF's internal CHECK assertions to fire → SIGABRT.  With 'spawn' each
        # worker starts a fresh Python interpreter with no inherited CUDA state.
        # _DMCObsWrapper must be a module-level class (not a local class) so
        # that Python's standard pickle can serialise it for the spawned workers.
        self.train_loader = DataLoader(
            _DMCObsWrapper(train_chain),
            batch_size=batch_size,
            num_workers=workers_per_gpu,
            pin_memory=pin_mem,
            multiprocessing_context="spawn",
        )
        self.val_loader = DataLoader(
            _DMCObsWrapper(val_chain),
            batch_size=val_batch,
            num_workers=workers_per_gpu,
            pin_memory=pin_mem,
            multiprocessing_context="spawn",
        )

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.val_loader   = self.accelerator.prepare(self.val_loader)

        self.train_dl_iter = cycle(self.train_loader)
        self.val_dl_iter   = cycle(self.val_loader)

    def _setup_multidataset_loaders(self):
        """
        Setup multi-dataset loaders from Hydra config.
        
        Expects config structure at self.cfg.data.multidataset:
            dataset_names: [list of dataset names]
            ratios: [list of ratios]
            batch_size: int
            seq_len: int
            num_workers: int
            pin_memory: bool
            shuffle_train: bool
            seed: int
            transforms: {resize: [H, W], normalize: {mean, std}}
            ssv2: {...}
            openx: {...}
            nwm: {...}
            loopnav: {...}
        """
        # Get multidataset config from Hydra
        md_cfg = self.cfg.data.multidataset
        
        # Convert OmegaConf to dict for compatibility with existing functions
        config = {
            "data": {
                "dataset_names": list(md_cfg.dataset_names),
                "ratios": list(md_cfg.ratios),
                "batch_size": md_cfg.batch_size,
                "val_batch_size": md_cfg.val_batch_size,
                "seq_len": md_cfg.seq_len,
                "num_workers": md_cfg.num_workers,
                "pin_memory": md_cfg.pin_memory,
                "shuffle_train": md_cfg.shuffle_train,
                "seed": md_cfg.seed,
            }
        }
        
        # Add dataset-specific configs if they exist
        if hasattr(md_cfg, 'procgen') and md_cfg.procgen is not None:
            config["data"]["procgen"] = OmegaConf.to_container(md_cfg.procgen, resolve=True)

        if hasattr(md_cfg, 'ssv2') and md_cfg.ssv2 is not None:
            config["data"]["ssv2"] = OmegaConf.to_container(md_cfg.ssv2, resolve=True)
        
        if hasattr(md_cfg, 'openx') and md_cfg.openx is not None:
            config["data"]["openx"] = OmegaConf.to_container(md_cfg.openx, resolve=True)
        
        if hasattr(md_cfg, 'nwm') and md_cfg.nwm is not None:
            config["data"]["nwm"] = OmegaConf.to_container(md_cfg.nwm, resolve=True)
        
        if hasattr(md_cfg, 'loopnav') and md_cfg.loopnav is not None:
            config["data"]["loopnav"] = OmegaConf.to_container(md_cfg.loopnav, resolve=True)

        if hasattr(md_cfg, 'memorymaze') and md_cfg.memorymaze is not None:
            config["data"]["memorymaze"] = OmegaConf.to_container(md_cfg.memorymaze, resolve=True)

        # Filter datasets based on data availability
        orig_names = list(config["data"]["dataset_names"])
        orig_ratios = list(config["data"]["ratios"])
        assert len(orig_names) == len(orig_ratios), (
            f"dataset_names and ratios must have same length, "
            f"got {len(orig_names)} and {len(orig_ratios)}"
        )

        filtered_names = []
        filtered_ratios = []
        
        for name, ratio in zip(orig_names, orig_ratios):
            # Check if required data exists for this logical dataset
            should_include = True

            if name == "procgen":
                if "procgen" not in config["data"]:
                    logger.warning(f"[skip] ProcGen config not found in multidataset config")
                    should_include = False
                elif not Path(config["data"]["procgen"]["data_dir"]).exists():
                    logger.warning(f"[skip] ProcGen data_dir not found: {config['data']['procgen']['data_dir']}")
                    should_include = False
            
            elif name == "ssv2":
                if "ssv2" not in config["data"]:
                    logger.warning(f"[skip] SSV2 config not found in multidataset config")
                    should_include = False
                elif not Path(config["data"]["ssv2"]["data_dir"]).exists():
                    logger.warning(f"[skip] SSV2 data_dir not found: {config['data']['ssv2']['data_dir']}")
                    should_include = False
            
            elif name == "openx":
                if "openx" not in config["data"]:
                    logger.warning(f"[skip] OpenX config not found in multidataset config")
                    should_include = False
                elif not Path(config["data"]["openx"]["data_dir"]).exists():
                    logger.warning(f"[skip] OpenX data_dir not found: {config['data']['openx']['data_dir']}")
                    should_include = False
            
            elif name == "nwm":
                if "nwm" not in config["data"]:
                    logger.warning(f"[skip] NWM config not found in multidataset config")
                    should_include = False
                elif not Path(config["data"]["nwm"]["data_dir"]).exists():
                    logger.warning(f"[skip] NWM data_dir not found: {config['data']['nwm']['data_dir']}")
                    should_include = False
            
            elif name == "loopnav":
                if "loopnav" not in config["data"]:
                    logger.warning(f"[skip] LoopNav config not found in multidataset config")
                    should_include = False
                elif not Path(config["data"]["loopnav"]["data_dir"]).exists():
                    logger.warning(f"[skip] LoopNav data_dir not found: {config['data']['loopnav']['data_dir']}")
                    should_include = False
            
            elif name == "memorymaze":
                if "memorymaze" not in config["data"]:
                    logger.warning(f"[skip] Memory Maze config not found in multidataset config")
                    should_include = False
                elif not Path(config["data"]["memorymaze"]["data_dir"]).exists():
                    logger.warning(f"[skip] Memory Maze data_dir not found: {config['data']['memorymaze']['data_dir']}")
                    should_include = False
            
            if should_include:
                filtered_names.append(name)
                filtered_ratios.append(ratio)
        
        if not filtered_names:
            raise ValueError("No datasets enabled. Please check your config and data paths.")
        
        config["data"]["dataset_names"] = filtered_names
        config["data"]["ratios"] = filtered_ratios
        
        logger.info(f"Enabled datasets: {config['data']['dataset_names']}")
        logger.info(f"Base ratios (per logical dataset): {config['data']['ratios']}")
        
        # Build dataset configs from logical datasets
        configs = create_dataset_configs_from_config(config)

        # Expand ratios to match the concrete dataset configs.
        # For 'procgen', we split its ratio equally across all specified games.
        expanded_ratios = []
        for name, ratio in zip(config["data"]["dataset_names"], config["data"]["ratios"]):
            if name == "procgen":
                procgen_cfg = config["data"]["procgen"]
                game_names = procgen_cfg.get("dataset_names", [])
                n_games = len(game_names)
                if n_games == 0:
                    logger.warning("ProcGen dataset_names is empty; skipping ProcGen datasets.")
                    continue
                per_game_ratio = ratio / float(n_games)
                expanded_ratios.extend([per_game_ratio] * n_games)
            else:
                expanded_ratios.append(ratio)

        if len(expanded_ratios) != len(configs):
            raise ValueError(
                f"Expanded ratios length {len(expanded_ratios)} does not match "
                f"number of dataset configs {len(configs)}. "
                "Please check your multidataset configuration (especially ProcGen)."
            )

        logger.info(f"Expanded ratios (per dataset config): {expanded_ratios}")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        self.train_loader = create_mixed_dataloader(
            configs=configs,
            ratios=expanded_ratios,
            batch_size=config["data"]["batch_size"],
            seq_len=config["data"]["seq_len"],
            transform=transform,
            mode="train",
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
            shuffle=config["data"].get("shuffle_train", True),
            seed=config["data"].get("seed", 42),
            logger=logger
        )

        self.val_loader = create_mixed_dataloader(
            configs=configs,
            ratios=expanded_ratios,
            batch_size=config["data"]["val_batch_size"],
            seq_len=config["data"]["seq_len"],
            transform=transform,
            mode="test",
            num_workers=config["data"]["num_workers"],
            pin_memory=config["data"]["pin_memory"],
            shuffle=False,
            seed=config["data"].get("seed", 42),
            logger=logger
        )

        self.train_loader = self.accelerator.prepare(self.train_loader)
        self.val_loader = self.accelerator.prepare(self.val_loader)

        self.train_dl_iter = cycle(self.train_loader)
        self.val_dl_iter = cycle(self.val_loader)

    def _augment_batch(self, x):
        """
        Apply consistent augmentations to video clips.
        x: [B, T, C, H, W]
        Returns: augmented x
        """
        B, T, C, H, W = x.shape
        aug_x = torch.empty_like(x)
        
        for b in range(B):
            # Random Resized Crop parameters
            # scale=(0.8, 1.0) to maintain most content but shift viewpoint
            # Using dummy tensor for get_params
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                torch.zeros(3, H, W), scale=(0.8, 1.0), ratio=(0.9, 1.1)
            )
            # Apply consistent crop to all frames in the sequence
            # x[b]: [T, C, H, W]
            aug_x[b] = TF.resized_crop(x[b], i, j, h, w, size=(H, W))
            
        return aug_x

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_distributed(self):
        return not (self.accelerator.distributed_type == DistributedType.NO and self.accelerator.num_processes == 1)

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def save_reconstruction_grid(self, gt_videos: torch.Tensor, recon_videos: torch.Tensor, current_train_step: int):
        if not self.is_main:
            return

        num_saved, T_recon, C_img, H, W = gt_videos.shape

        gt_videos = torch.clamp(gt_videos, 0.0, 1.0)
        recon_videos = torch.clamp(recon_videos, 0.0, 1.0)

        output_strips = []
        for i in range(num_saved):
            # Concatenate all time steps horizontally for GT
            gt_strip = torch.cat([gt_videos[i, t] for t in range(T_recon)], dim=2)  # (C, H, T*W)
            # Concatenate all time steps horizontally for Recon
            recon_strip = torch.cat([recon_videos[i, t] for t in range(T_recon)], dim=2)  # (C, H, T*W)
            # Stack GT strip above Recon strip
            comparison_strip = torch.cat((gt_strip, recon_strip), dim=1)  # (C, 2*H, T*W)
            output_strips.append(comparison_strip)

        # Make a grid of these comparison strips (num_saved rows, 1 col)
        # Add padding between samples if desired
        grid = torchvision.utils.make_grid(output_strips, nrow=1, padding=5, pad_value=0.5)

        try:
            save_path = self.work_dir / f"reconstruction/reconstructions_step_{current_train_step}-{self.exp_name}.png"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torchvision.utils.save_image(grid, save_path)
            logger.info(f"Saved reconstruction grid to {save_path}")

        except Exception as e:
            logger.error(f"Error saving reconstruction grid: {e}")

    def load(self, path: str):
        p = Path(path)
        if not p.exists():
            logger.info(f"Checkpoint not found at {str(p)}, starting from scratch.")
            return

        try:
            logger.info(f"Loading checkpoint from {str(p)}...")
            data = torch.load(p, map_location="cpu")

            model_to_load = self.accelerator.unwrap_model(self.model)

            # Handle potential mismatch if model was saved directly vs. via get_state_dict
            if "model" in data:
                model_state = data["model"]
            elif "module" in data:  # If it was a DDP model state_dict
                model_state = data["module"]
            else:
                model_state = data  # Assume entire data is model state if 'model' key is missing

            msg = model_to_load.load_state_dict(model_state, strict=False)
            logger.info(f"Model loaded with message: {msg}")

            if "optimizer" in data:
                self.optimizer.load_state_dict(data["optimizer"])
            else:
                logger.info("Warning: Optimizer state not found in checkpoint.")

            self.current_step = int(data.get("steps", data.get("step", 0))) + 1
            self.current_val_step = int(
                data.get("val_steps", data.get("val_step", self.current_step // self.val_every_n_steps))
            )

            logger.info(f"Resumed training from checkpoint {str(p)} at step {self.current_step}")

        except Exception as e:
            logger.error(f"Failed to load checkpoint from {str(p)}: {e}. Starting from scratch.")
            self.current_step = 0

    def save(self, path: str, is_milestone: bool = False):
        if not self.is_main:
            return

        p = Path(path)
        logger.info(f"Saving checkpoint to {str(p)} at step {self.current_step}...")

        save_data = {
            "model": self.accelerator.get_state_dict(self.model),
            "steps": self.current_step,
            "val_steps": self.current_val_step,
        }

        if not is_milestone or (is_milestone and self.milestone_optim_state):
            save_data["optimizer"] = self.optimizer.state_dict()
            save_data["scheduler"] = self.scheduler.state_dict()

        tmp_path = None
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            # Safe atomic overwrite for non-milestone checkpoints
            tmp_path = p.with_suffix(p.suffix + ".tmp")

            self.accelerator.save(save_data, tmp_path)  # Write to temp first
            tmp_path.replace(p)  # Atomically replace existing file

            logger.info(f"Checkpoint saved successfully to {str(p)}.")
        except Exception as e:
            logger.error(f"Failed to save checkpoint to {str(p)}: {e}")
            if tmp_path is not None and tmp_path.exists():
                tmp_path.unlink()  # Clean up partial file

    def train_step(self):
        self.model.train()
        total_loss_value_accum = 0.0

        for i in range(self.grad_accum_every):
            is_last_accum_step = i == self.grad_accum_every - 1

            with self.accelerator.accumulate(self.model):
                batch_data = next(self.train_dl_iter)
                videos = batch_data.get("observations", batch_data.get("images")).to(self.device)
                actions = batch_data.get("action", None)
                if actions is not None:
                    actions = actions.to(self.device)
                if self.phase == 3 and actions is None:
                    raise ValueError(
                        "Phase 3 requires batches with an `action` field. "
                        "Use an action-labeled dataset such as `data=robotask`."
                    )

                batch_size, seq_len, *_ = videos.size()

                videos_flat = videos.flatten(0, 1)  # [b*t, 3, 256, 256]
                X_enc = self.rae.encode(videos_flat)
                X_enc = rearrange(X_enc, '(b t) c h w -> b t c h w', b=batch_size, t=seq_len)
                loss, loss_dict = self.model(X_enc, actions=actions) # Use encoded X

                loss_to_backward = loss / self.grad_accum_every
                self.accelerator.backward(loss_to_backward)

                total_loss_value_accum += loss_to_backward.item()

        grad_norm_val = 0.0
        if self.max_grad_norm is not None:
            grad_norm_val = self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            grad_norm_val = grad_norm_val.item()

        self.optimizer.step()

        # Logging
        log_payload = {"step": self.current_step}
        # Add all scalar losses from loss_dict
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor) and value.numel() == 1:
                log_payload[key] = value.item()
            elif isinstance(value, (int, float)):
                log_payload[key] = value

        # Add accumulated loss if not already in loss_dict or if different
        if "total_loss_accumulated_step" not in log_payload:  # Use a distinct name
            log_payload["total_loss_accumulated_step"] = total_loss_value_accum

        # Parameter and Gradient Norms
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        param_norm = torch.norm(
            torch.stack([torch.norm(p.detach().float(), 2.0) for p in unwrapped_model.parameters() if p.requires_grad])
        ).item()

        log_payload["param_norm"] = param_norm
        log_payload["grad_norm"] = grad_norm_val

        self.optimizer.zero_grad()

        return total_loss_value_accum, log_payload

    @torch.no_grad()
    def run_validation_and_log(self, train_step: int):
        if self.val_dl_iter is None:
            return {}

        logger.info(f"Running validation step {self.current_val_step} at training step {train_step}...")
        model_for_eval = self.accelerator.unwrap_model(self.model)
        model_for_eval.eval()

        total_val_loss = 0.0
        all_val_logs = {}  # To average all reported losses

        first_batch_videos = None
        first_batch_recons = None

        batch_idx = 0
        while batch_idx < self.num_val_batches_to_log:
            val_batch_data = next(self.val_dl_iter)
            videos = val_batch_data.get("observations", val_batch_data.get("images")).to(self.device)
            actions = val_batch_data.get("action", None)
            if actions is not None:
                actions = actions.to(self.device)
            if self.phase == 3 and actions is None:
                raise ValueError(
                    "Phase 3 validation requires batches with an `action` field. "
                    "Use an action-labeled validation dataset such as `data=robotask`."
                )
            batch_size, seq_len, *_ = videos.size()
            videos_flat = videos.flatten(0, 1)  # [b*t, 3, 256, 256]
            X_enc = self.rae.encode(videos_flat)
            X_enc = rearrange(X_enc, '(b t) c h w -> b t c h w', b=batch_size, t=seq_len)
            val_loss, val_loss_dict = model_for_eval(X_enc, actions=actions)

            # Store first batch for visualization
            if batch_idx == 0 and self.num_val_samples_to_save > 0:
                # res_dict = model_for_eval.one_step_forward(X_enc)
                latent_actions = model_for_eval.get_latent_actions(X_enc)
                res_dict = model_for_eval.autoregressive_forward(X_enc[:, :1], latent_actions)

                # actions = actions[:, 1:-1]
                # batch_size, seq_len, *_ = actions.size()
                # actions = actions.flatten(0, 1)
                # z = model_for_eval.Action_decoder(actions)
                # z = rearrange(z, '(b t) d -> b t d', b=batch_size, t=seq_len)
                # res_dict = model_for_eval.prediction(X_enc[:, :2], z)
                
                embedding_gen_flat = res_dict['embedding_gen'].flatten(0, 1)
                recon_videos = self.rae.decode(embedding_gen_flat)
                recon_videos = rearrange(recon_videos, '(b t) c h w -> b t c h w', b=batch_size, t=seq_len-1)
                # recon_videos = rearrange(recon_videos, '(b t) c h w -> b t c h w', b=batch_size, t=seq_len)

                batch_size = videos.shape[0]
                num_to_save = min(batch_size, self.num_val_samples_to_save)

                # Randomly permute the indices (on the same device as videos)
                indices = torch.randperm(batch_size, device=videos.device)[:num_to_save]

                # Use the shuffled indices to index into your tensors
                first_batch_videos = videos[indices, ...].cpu()
                first_batch_recons = torch.cat([videos[indices, 0:1, ...], recon_videos[indices, ...]], dim=1).cpu()

            for key, value in val_loss_dict.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    all_val_logs[key] = all_val_logs.get(key, 0.0) + value.item()
                elif isinstance(value, (int, float)):
                    all_val_logs[key] = all_val_logs.get(key, 0.0) + value
            total_val_loss += val_loss.item()

            batch_idx += 1

        # Average losses
        avg_val_logs = {}
        if batch_idx > 0:  # Ensure at least one batch was processed
            for key, value in all_val_logs.items():
                avg_val_logs[f"val/{key}"] = value / batch_idx
            avg_val_logs["val/total_loss_avg"] = total_val_loss / batch_idx
        avg_val_logs["val/train_step"] = train_step
        avg_val_logs["val/val_step"] = self.current_val_step

        logger.info(f"Validation at step {train_step}: {avg_val_logs}")

        # Save reconstruction grid
        if first_batch_videos is not None and first_batch_recons is not None:
            self.save_reconstruction_grid(first_batch_videos, first_batch_recons, train_step)

        self.current_val_step += 1
        return avg_val_logs

    def run(self):
        best_loss = float('inf')
        checkpoint_path = Path(self.work_dir)
        # Ensure checkpoint path exists (only on main process)
        if self.accelerator.is_main_process:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
        self.accelerator.wait_for_everyone()

        logger.info(f"Starting training from step {self.current_step} up to {self.num_train_steps} steps.")
        checkpoint_path = Path(self.work_dir) / f'phase{self.phase}'
        while self.current_step < self.num_train_steps:
            avg_loss_this_step, logs_train = self.train_step()

            logger.info(f"Step {self.current_step}/{self.num_train_steps}: {logs_train}")

            # Validation must run on main process
            logs_val = {}
            if self.current_step % self.val_every_n_steps == 0:
                logs_val = self.run_validation_and_log(self.current_step)

            # Update schedulers
            # self.scheduler.step()

            if self.is_main:
                # Overwrite the latest checkpoint
                if self.current_step % self.save_model_every == 0:
                    self.save(checkpoint_path / f"{self.exp_name}.pt")

                # Milestone: keep permanent
                if self.current_step % self.save_milestone_every == 0:
                    self.save(checkpoint_path / f"{self.exp_name}_milestone_{self.current_step}.pt", is_milestone=True)
            # merge logs for training and validation
            if len(logs_val) > 0:
                logs = {**logs_train, **logs_val}
            else:
                logs = logs_train
            self.accelerator.log(logs)

            self.current_step += 1
            self.accelerator.wait_for_everyone()

        if self.is_main:
            self.save(checkpoint_path / f"{self.exp_name}_final.pt", is_milestone=True)
            logger.info("Training complete.")
            if self.accelerator.trackers:
                self.accelerator.end_training()
