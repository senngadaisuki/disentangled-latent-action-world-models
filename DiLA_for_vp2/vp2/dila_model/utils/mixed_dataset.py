"""
Elegant Mixed Dataset Implementation for Multi-Dataset Training

This module provides a clean, modular approach to mixing multiple datasets for training.
Key features:
- Configuration-driven dataset paths and parameters
- Registry pattern for easy dataset addition
- Efficient batch sampling without nested DataLoaders
- Clean separation of concerns
"""

import torch
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod

# ProcGen officially exposes 16 environments; allow any of them to be mixed.
PROCGEN_ENV_NAMES = [
    "bigfish",
    "bossfight",
    "caveflyer",
    "chaser",
    "climber",
    "coinrun",
    "dodgeball",
    "fruitbot",
    "heist",
    "jumper",
    "leaper",
    "maze",
    "miner",
    "ninja",
    "plunder",
    "starpilot",
]


@dataclass
class DatasetConfig:
    """Configuration for a single dataset"""
    name: str
    data_dir: str = ""
    train_file: str = ""
    val_file: str = ""
    frame_step: int = 8
    additional_params: Optional[Dict[str, Any]] = None


class DatasetFactory(ABC):
    """Abstract factory for creating datasets"""
    
    @abstractmethod
    def create_train_dataset(self, config: DatasetConfig, seq_len: int, 
                           transform: Optional[Callable]) -> Dataset:
        """Create training dataset"""
        pass
    
    @abstractmethod
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        """Create validation dataset"""
        pass
    
    @abstractmethod
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """Normalize batch output to standard format (B, T, C, H, W)"""
        pass


# class ProcGenFactory(DatasetFactory):
#     """Factory for ProcGen datasets (miner, maze)"""
    
#     def __init__(self):
#         from .ProcGen import data_loader
#         self.data_loader = data_loader
    
#     def create_train_dataset(self, config: DatasetConfig, seq_len: int, 
#                            transform: Optional[Callable]) -> Dataset:
#         dataset = self.data_loader.lazy_load(
#             config.name, seq_len=seq_len, transform=transform, overlap=False, 
#             train=True
#         )
#         return dataset
    
#     def create_val_dataset(self, config: DatasetConfig, seq_len: int,
#                           transform: Optional[Callable]) -> Dataset:
#         dataset = self.data_loader.lazy_load(
#             config.name, seq_len=seq_len, transform=transform, overlap=False,
#             train=False
#         )
#         return dataset
    
#     def normalize_batch(self, batch: Any) -> torch.Tensor:
#         """ProcGen uses TensorDict with 'obs' key"""
#         return batch["obs"]


class ProcGenFactory(DatasetFactory):
    def create_train_dataset(self, config: DatasetConfig, seq_len: int,
                           transform: Optional[Callable]) -> Dataset:
        from utils.datasets import ProcGenDataset
        
        params = config.additional_params or {}
        stride = params.get('stride', seq_len)
        split_ratio = params.get('split_ratio', 0.8)
        seed = params.get('seed', 42)
        return_goal = params.get('return_goal', True)
        
        return ProcGenDataset(
            root_dir=config.data_dir + '/' + config.name,
            seq_len=seq_len,
            stride=stride,
            transform=transform,
            split_ratio=split_ratio,
            mode='train',
            seed=seed,
            return_goal=return_goal,
        )
    
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        from utils.datasets import ProcGenDataset
        
        params = config.additional_params or {}
        stride = params.get('stride', seq_len)
        split_ratio = params.get('split_ratio', 0.8)
        seed = params.get('seed', 42)
        return_goal = params.get('return_goal', True)
        
        return ProcGenDataset(
            root_dir=config.data_dir + '/' + config.name,
            seq_len=seq_len,
            stride=stride,
            transform=transform,
            split_ratio=split_ratio,
            mode='val',
            seed=seed,
            return_goal=return_goal,
        )
    
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """
        LoopNav returns a dict with 'images' key containing [seq_len, C, H, W] tensor.
        """
        if isinstance(batch, dict):
            return batch['images']
        return batch


class DynamoFactory(DatasetFactory):
    """Factory for Dynamo datasets (sim_kitchen, libero, block_pushing, pusht)"""
    
    def __init__(self, dataset_class: type):
        self.dataset_class = dataset_class
    
    def create_train_dataset(self, config: DatasetConfig, seq_len: int,
                           transform: Optional[Callable]) -> Dataset:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
        from dynamo.datasets.core import get_train_val_sliced
        from dynamo.datasets.utils import Adapted_Transform
        
        dataset = self.dataset_class(data_directory=config.data_dir)
        adapted_transform = Adapted_Transform(transform) if transform else None
        train_dataset, _ = get_train_val_sliced(
            dataset, window_size=seq_len, transform=adapted_transform,
            random_seed=42, frame_step=config.frame_step
        )
        return train_dataset
    
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils'))
        from dynamo.datasets.core import get_train_val_sliced
        from dynamo.datasets.utils import Adapted_Transform
        
        dataset = self.dataset_class(data_directory=config.data_dir)
        adapted_transform = Adapted_Transform(transform) if transform else None
        _, val_dataset = get_train_val_sliced(
            dataset, window_size=seq_len, transform=adapted_transform,
            random_seed=42, frame_step=config.frame_step
        )
        return val_dataset
    
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """Dynamo datasets return tuple, take first view"""
        return batch[0][:, :, 0]


class SSV2Factory(DatasetFactory):
    """Factory for Something-Something-V2 dataset"""
    def create_train_dataset(self, config: DatasetConfig, seq_len: int,
                           transform: Optional[Callable]) -> Dataset:
        # Import here to avoid circular dependency
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.datasets import SomethingSomethingV2Dataset
        
        params = config.additional_params or {}
        # Use seq_len as sliding_window to ensure output has T=seq_len frames
        frames_per_clip = params.get('frames_per_clip', seq_len)
        sliding_window = params.get('sliding_window', seq_len)
        sample_downsample_rate = params.get('sample_downsample_rate', 1)
        
        
        return SomethingSomethingV2Dataset(
            root_dir=config.data_dir,
            annotations_file=config.train_file,
            transform=transform,
            frames_per_clip=frames_per_clip,
            sliding_window=sliding_window,
            sample_downsample_rate=sample_downsample_rate
        )
    
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.datasets import SomethingSomethingV2Dataset
        
        params = config.additional_params or {}
        frames_per_clip = params.get('frames_per_clip', seq_len)
        sliding_window = params.get('sliding_window', seq_len)
        sample_downsample_rate = params.get('sample_downsample_rate', 1)
        
        return SomethingSomethingV2Dataset(
            root_dir=config.data_dir,
            annotations_file=config.val_file,
            transform=transform,
            frames_per_clip=frames_per_clip,
            sliding_window=sliding_window,
            sample_downsample_rate=sample_downsample_rate * 10
        )
    
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """
        SSV2 returns tensor with shape (num_clips, sliding_window, C, H, W) for single sample.
        Take the first clip to get (sliding_window, C, H, W) = (T, C, H, W).
        
        Note: batch here is a single sample, not a batched tensor!
        """
        # batch shape: [num_clips, sliding_window, C, H, W]
        # Take first clip: [sliding_window, C, H, W]
        return batch.flatten(0, 1)


# class OpenXEmbodimentFactory(DatasetFactory):
#     """Factory for Open-X-Embodiment dataset"""
    
#     def create_train_dataset(self, config: DatasetConfig, seq_len: int,
#                            transform: Optional[Callable]) -> Dataset:
#         from .openx_dataset import OpenXEmbodimentDataset
        
#         params = config.additional_params or {}
#         frame_skip = params.get('frame_skip', 1)
#         dataset_names = params.get('dataset_names', None)  # List of specific datasets to load
#         max_episodes_per_dataset = params.get('max_episodes_per_dataset', None)
        
#         return OpenXEmbodimentDataset(
#             data_dir=config.data_dir,
#             split='train',
#             seq_len=seq_len,
#             transform=transform,
#             frame_skip=frame_skip,
#             dataset_names=dataset_names,
#             max_episodes_per_dataset=max_episodes_per_dataset
#         )
    
#     def create_val_dataset(self, config: DatasetConfig, seq_len: int,
#                           transform: Optional[Callable]) -> Dataset:
#         from .openx_dataset import OpenXEmbodimentDataset
        
#         params = config.additional_params or {}
#         frame_skip = params.get('frame_skip', 1)
#         dataset_names = params.get('dataset_names', None)
#         max_episodes_per_dataset = params.get('max_episodes_per_dataset', None)
        
#         return OpenXEmbodimentDataset(
#             data_dir=config.data_dir,
#             split='val',
#             seq_len=seq_len,
#             transform=transform,
#             frame_skip=frame_skip,
#             dataset_names=dataset_names,
#             max_episodes_per_dataset=max_episodes_per_dataset
#         )
    
#     def normalize_batch(self, batch: Any) -> torch.Tensor:
#         """Open-X-Embodiment returns dict with 'observation/image' key"""
#         if isinstance(batch, dict):
#             return batch.get('observation/image', batch.get('image', batch.get('observations')))
#         return batch


class OpenXEmbodimentFactory(DatasetFactory):
    """
    Factory for Open-X-Embodiment datasets that have been pre-processed 
    into folder + frames format using openx_process.py.
    
    Expected data structure:
        data_dir/
        ├── {dataset_name}_videos/
        │   ├── {dataset_name}.txt
        │   ├── video_0/
        │   │   ├── frame_0.png
        │   │   └── ...
        │   └── ...
    """
    
    def create_train_dataset(self, config: DatasetConfig, seq_len: int,
                           transform: Optional[Callable]) -> Dataset:
        from utils.datasets import OpenXProcessedDataset
        
        params = config.additional_params or {}
        frame_skip = params.get('frame_skip', 1)
        dataset_names = params.get('dataset_names', None)
        max_episodes_per_dataset = params.get('max_episodes_per_dataset', None)
        train_ratio = params.get('train_ratio', 0.9)
        seed = params.get('seed', 42)
        return_instruction = params.get('return_instruction', False)
        # Memory optimization: use lazy loading by default
        lazy_frame_list = params.get('lazy_frame_list', True)
        
        return OpenXProcessedDataset(
            data_dir=config.data_dir,
            split='train',
            seq_len=seq_len,
            transform=transform,
            frame_skip=frame_skip,
            dataset_names=dataset_names,
            max_episodes_per_dataset=max_episodes_per_dataset,
            train_ratio=train_ratio,
            seed=seed,
            return_instruction=return_instruction,
            lazy_frame_list=lazy_frame_list
        )
    
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        from utils.datasets import OpenXProcessedDataset
        
        params = config.additional_params or {}
        frame_skip = params.get('frame_skip', 1)
        dataset_names = params.get('dataset_names', None)
        max_episodes_per_dataset = params.get('max_episodes_per_dataset', None)
        train_ratio = params.get('train_ratio', 0.9)
        seed = params.get('seed', 42)
        return_instruction = params.get('return_instruction', False)
        # Memory optimization: use lazy loading by default
        lazy_frame_list = params.get('lazy_frame_list', True)
        
        return OpenXProcessedDataset(
            data_dir=config.data_dir,
            split='val',
            seq_len=seq_len,
            transform=transform,
            frame_skip=frame_skip,
            dataset_names=dataset_names,
            max_episodes_per_dataset=max_episodes_per_dataset,
            train_ratio=train_ratio,
            seed=seed,
            return_instruction=return_instruction,
            lazy_frame_list=lazy_frame_list
        )
    
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """
        OpenX processed dataset returns tensor directly or dict with 'observations'.
        """
        if isinstance(batch, dict):
            return batch.get('observations', batch.get('image', batch))
        return batch


class NWMFactory(DatasetFactory):
    """Factory for Navigation with Memory datasets (NWM / NoMaD style)"""
    
    def _build_kwargs(self, config: DatasetConfig, seq_len: int, transform: Optional[Callable]) -> Dict[str, Any]:
        params = config.additional_params or {}
        
        # Get goals_per_obs first (default 1)
        goals_per_obs = params.get("goals_per_obs", 1)
        
        # Calculate context_size to make total output length = seq_len
        # NWM returns obs_image with shape [context_size + goals_per_obs, C, H, W]
        # So context_size = seq_len - goals_per_obs
        default_context_size = max(1, seq_len - goals_per_obs)
        context_size = params.get("context_size", default_context_size)
        
        return {
            "data_folder": config.data_dir,
            "data_split_folder": params.get("data_split_folder", config.data_dir),
            "dataset_name": params.get("dataset_name", "recon"),
            "image_size": tuple(params.get("image_size", (128, 128))),
            "min_dist_cat": params.get("min_dist_cat", 3),
            "max_dist_cat": params.get("max_dist_cat", 10),
            "len_traj_pred": params.get("len_traj_pred", seq_len),
            "traj_stride": params.get("traj_stride", 1),
            "context_size": context_size,
            "transform": params.get("transform_override", None) or transform,
            "traj_names": params.get("traj_names", "traj_names.txt"),
            "normalize": params.get("normalize", True),
            "predefined_index": params.get("predefined_index"),
            "goals_per_obs": goals_per_obs,
        }
    
    def create_train_dataset(self, config: DatasetConfig, seq_len: int,
                           transform: Optional[Callable]) -> Dataset:
        from utils.datasets import TrainingDataset
        kwargs = self._build_kwargs(config, seq_len, transform)
        kwargs['data_split_folder'] = kwargs['data_split_folder'] + '/train/'
        return TrainingDataset(**kwargs)
    
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        from utils.datasets import EvalDataset
        kwargs = self._build_kwargs(config, seq_len, transform)
        kwargs['data_split_folder'] = kwargs['data_split_folder'] + '/test/'
        return EvalDataset(**kwargs)
    
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """
        NWM datasets return tuples:
        - TrainingDataset: (obs_image, goal_pos, rel_time)
          - obs_image shape: [context_size + goals_per_obs, C, H, W]
        - EvalDataset: (idx, obs_image, pred_image, delta)
          - pred_image shape: [len_traj_pred, C, H, W]
        
        We extract the image tensor and return it as (T, C, H, W).
        """
        if isinstance(batch, (list, tuple)):
            # Check if it's EvalDataset by looking at first element (should be index tensor)
            if len(batch) >= 3 and isinstance(batch[0], torch.Tensor) and batch[0].numel() == 1:
                # EvalDataset: (idx, obs_image, pred_image, delta)
                return batch[2]  # pred_image has more frames
            elif len(batch) >= 1:
                # TrainingDataset: (obs_image, goal_pos, rel_time)
                return batch[0]  # obs_image
        return batch


class LoopNavFactory(DatasetFactory):
    """
    Factory for LoopNav dataset (Minecraft navigation dataset).
    
    LoopNav returns a dict with:
        - 'images': [seq_len, C, H, W]
        - 'x', 'y', 'z': [seq_len] positions
        - 'yaw', 'pitch': [seq_len] orientations
        - 'action': [seq_len, 4] actions
        - 'goal': [seq_len, 3] goal positions (optional)
        - 'path': relative path string
    """
    
    def create_train_dataset(self, config: DatasetConfig, seq_len: int,
                           transform: Optional[Callable]) -> Dataset:
        from utils.datasets import LoopNavDataset
        
        params = config.additional_params or {}
        stride = params.get('stride', seq_len)
        split_ratio = params.get('split_ratio', 0.8)
        seed = params.get('seed', 42)
        return_goal = params.get('return_goal', True)
        # Memory optimization: disable caching by default
        cache_metadata = params.get('cache_metadata', False)
        max_cached_meta = params.get('max_cached_meta', 1000)
        
        return LoopNavDataset(
            root_dir=config.data_dir,
            seq_len=seq_len,
            stride=stride,
            transform=transform,
            split_ratio=split_ratio,
            mode='train',
            seed=seed,
            return_goal=return_goal,
            cache_metadata=cache_metadata,
            max_cached_meta=max_cached_meta
        )
    
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        from utils.datasets import LoopNavDataset
        
        params = config.additional_params or {}
        stride = params.get('stride', seq_len)
        split_ratio = params.get('split_ratio', 0.8)
        seed = params.get('seed', 42)
        return_goal = params.get('return_goal', True)
        # Memory optimization: disable caching by default
        cache_metadata = params.get('cache_metadata', False)
        max_cached_meta = params.get('max_cached_meta', 1000)
        
        return LoopNavDataset(
            root_dir=config.data_dir,
            seq_len=seq_len,
            stride=stride,
            transform=transform,
            split_ratio=split_ratio,
            mode='val',
            seed=seed,
            return_goal=return_goal,
            cache_metadata=cache_metadata,
            max_cached_meta=max_cached_meta
        )
    
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """
        LoopNav returns a dict with 'images' key containing [seq_len, C, H, W] tensor.
        """
        if isinstance(batch, dict):
            return batch['images']
        return batch


class DMCVisionFactory(DatasetFactory):
    """
    Factory for DMC Vision Benchmark (DMC-VB) dataset.

    DMC-VB is stored in RLDS/TFDS format.  Each sample is a dict with::

        'images': Tensor[T, C, H, W]  float32  (pixel values in obs_vrange)

    Supported domains: 'walker', 'cheetah', 'humanoid', 'ant'.

    NOTE: DMCVisionDataset is an IterableDataset (streaming, lazy loading).
    It is NOT compatible with MixedDataset, which requires map-style
    Datasets with __getitem__ and __len__.  Use it standalone via a plain
    DataLoader instead.

    Expected config keys (all optional with sensible defaults):
        domain_name, policy_level, difficulty, dynamic_distractors,
        img_height, img_width, obs_vrange, target_hidden, ant_fixed_seed,
        max_samples, stride, shuffle, shuffle_buffer_size,
        train_split, val_split
    """

    def _build_kwargs(
        self, config: DatasetConfig, seq_len: int, transform, split: str
    ) -> dict:
        params = config.additional_params or {}
        return dict(
            data_dir=config.data_dir,
            domain_name=params.get('domain_name', 'walker'),
            task_name=params.get('task_name', None),
            policy_level=params.get('policy_level', 'expert'),
            difficulty=params.get('difficulty', 'none'),
            dynamic_distractors=params.get('dynamic_distractors', False),
            seq_len=seq_len,
            split=split,
            img_height=params.get('img_height', 64),
            img_width=params.get('img_width', 64),
            obs_vrange=tuple(params.get('obs_vrange', [0.0, 1.0])),
            transform=transform,
            target_hidden=params.get('target_hidden', False),
            ant_fixed_seed=params.get('ant_fixed_seed', None),
            max_samples=params.get('max_samples', None),
            stride=params.get('stride', None),
            shuffle=params.get('shuffle', False),
            shuffle_buffer_size=params.get('shuffle_buffer_size', 1000),
        )

    def create_train_dataset(
        self, config: DatasetConfig, seq_len: int, transform
    ) -> Dataset:
        from utils.datasets_dmc import DMCVisionDataset
        params = config.additional_params or {}
        train_split = params.get('train_split', 'train')
        kwargs = self._build_kwargs(config, seq_len, transform, train_split)
        return DMCVisionDataset(**kwargs)

    def create_val_dataset(
        self, config: DatasetConfig, seq_len: int, transform
    ) -> Dataset:
        from utils.datasets_dmc import DMCVisionDataset
        params = config.additional_params or {}
        val_split = params.get('val_split', 'test')
        kwargs = self._build_kwargs(config, seq_len, transform, val_split)
        return DMCVisionDataset(**kwargs)

    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """DMCVisionDataset returns a dict with 'images' key."""
        if isinstance(batch, dict):
            return batch['images']
        return batch


class MemoryMazeFactory(DatasetFactory):
    """
    Factory for Memory Maze dataset.
    
    Memory Maze returns a dict with:
        - 'images': [seq_len, C, H, W]
        - 'action': [seq_len, action_dim] actions
        - 'goal': [seq_len, 3] goal positions (optional)
        - 'state': [seq_len, 5] state (x, y, z, yaw, pitch) (optional)
        - 'path': relative path string
    """
    
    def create_train_dataset(self, config: DatasetConfig, seq_len: int,
                           transform: Optional[Callable]) -> Dataset:
        from utils.trajectorydataset import TrajectoryDataset
        
        params = config.additional_params or {}
        stride = params.get('stride', seq_len)
        split_ratio = params.get('split_ratio', 0.8)
        seed = params.get('seed', 42)
        return_goal = params.get('return_goal', False)
        return_state = params.get('return_state', False)
        cache_metadata = params.get('cache_metadata', False)
        
        return TrajectoryDataset(
            root_dir=config.data_dir,
            seq_len=seq_len,
            stride=stride,
            transform=transform,
            split_ratio=split_ratio,
            mode='train',
            seed=seed,
            return_goal=return_goal,
            cache_metadata=cache_metadata,
            return_state=return_state
        )
    
    def create_val_dataset(self, config: DatasetConfig, seq_len: int,
                          transform: Optional[Callable]) -> Dataset:
        from utils.trajectorydataset import TrajectoryDataset
        
        params = config.additional_params or {}
        stride = params.get('stride', seq_len)
        split_ratio = params.get('split_ratio', 0.8)
        seed = params.get('seed', 42)
        return_goal = params.get('return_goal', False)
        return_state = params.get('return_state', False)
        cache_metadata = params.get('cache_metadata', False)
        
        return TrajectoryDataset(
            root_dir=config.data_dir,
            seq_len=seq_len,
            stride=stride,
            transform=transform,
            split_ratio=split_ratio,
            mode='test',  # TrajectoryDataset uses 'test' for validation
            seed=seed,
            return_goal=return_goal,
            cache_metadata=cache_metadata,
            return_state=return_state
        )
    
    def normalize_batch(self, batch: Any) -> torch.Tensor:
        """
        Memory Maze returns a dict with 'images' key containing [seq_len, C, H, W] tensor.
        """
        if isinstance(batch, dict):
            return batch['images']
        return batch


class DatasetRegistry:
    """
    Registry for dataset factories.
    
    Note: Uses class-level storage. Call clear() before re-initialization
    if you want to avoid memory accumulation across multiple runs.
    """
    
    _factories: Dict[str, DatasetFactory] = {}
    _initialized: bool = False
    
    @classmethod
    def register(cls, name: str, factory: DatasetFactory):
        """Register a dataset factory"""
        cls._factories[name] = factory
    
    @classmethod
    def get_factory(cls, name: str) -> DatasetFactory:
        """Get factory by name, creating a fresh instance to avoid state leakage"""
        if name not in cls._factories:
            raise ValueError(f"Dataset '{name}' not registered. Available: {list(cls._factories.keys())}")
        # Return a fresh factory instance to avoid shared state issues in multiprocessing
        factory_class = type(cls._factories[name])
        if factory_class == ProcGenFactory:
            return ProcGenFactory()
        elif factory_class == SSV2Factory:
            return SSV2Factory()
        elif factory_class == OpenXEmbodimentFactory:
            return OpenXEmbodimentFactory()
        elif factory_class == NWMFactory:
            return NWMFactory()
        elif factory_class == LoopNavFactory:
            return LoopNavFactory()
        elif factory_class == MemoryMazeFactory:
            return MemoryMazeFactory()
        elif factory_class == DMCVisionFactory:
            return DMCVisionFactory()
        return cls._factories[name]
    
    @classmethod
    def clear(cls):
        """Clear all registered factories to free memory"""
        cls._factories.clear()
        cls._initialized = False
    
    @classmethod
    def initialize_default_factories(cls):
        """Initialize all default dataset factories (idempotent)"""
        if cls._initialized:
            return
        
        # Register ProcGen datasets
        for env_name in PROCGEN_ENV_NAMES:
            cls.register(env_name, ProcGenFactory())
        # Register SSV2
        cls.register("ssv2", SSV2Factory())
        
        # Register Open-X-Embodiment (processed folder + frames format)
        cls.register("openx", OpenXEmbodimentFactory())

        # Register NWM
        cls.register("nwm", NWMFactory())

        # Register LoopNav
        cls.register("loopnav", LoopNavFactory())
        
        # Register Memory Maze
        cls.register("memorymaze", MemoryMazeFactory())

        # Register DMC Vision Benchmark
        cls.register("dmc_vision", DMCVisionFactory())

        cls._initialized = True


class MixedBatchSampler(Sampler):
    """
    Sampler that generates mixed batches from multiple datasets.
    
    This sampler generates indices for each dataset according to specified ratios,
    ensuring balanced sampling across datasets.
    """
    
    def __init__(
        self,
        dataset_lengths: List[int],
        ratios: List[float],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True
    ):
        """
        Args:
            dataset_lengths: List of dataset lengths
            ratios: List of sampling ratios for each dataset
            batch_size: Total batch size
            shuffle: Whether to shuffle samples
            seed: Random seed
            drop_last: Drop last incomplete batch
        """
        self.dataset_lengths = dataset_lengths
        self.ratios = ratios
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        
        # Calculate per-dataset batch sizes
        self.batch_sizes = self._calculate_batch_sizes()
        
        # Calculate total number of batches (limited by smallest dataset ratio)
        self.num_batches = self._calculate_num_batches()
        
    def _calculate_batch_sizes(self) -> List[int]:
        """Calculate how many samples from each dataset per batch"""
        total_ratio = sum(self.ratios)
        batch_sizes = []
        remaining = self.batch_size
        
        for i, ratio in enumerate(self.ratios):
            if i == len(self.ratios) - 1:
                # Last dataset gets remaining samples
                allocated = remaining
            else:
                allocated = max(1, round(self.batch_size * ratio / total_ratio))
                allocated = min(allocated, remaining - (len(self.ratios) - i - 1))
            batch_sizes.append(allocated)
            remaining -= allocated
        
        # Adjust if there's a mismatch
        diff = self.batch_size - sum(batch_sizes)
        if diff > 0:
            batch_sizes[0] += diff
        elif diff < 0:
            batch_sizes[0] += diff  # diff is negative
            
        return batch_sizes
    
    def _calculate_num_batches(self) -> int:
        """Calculate number of batches based on dataset sizes and ratios"""
        # Number of batches limited by how many complete batches we can form from each dataset
        max_batches_per_dataset = [
            length // bs for length, bs in zip(self.dataset_lengths, self.batch_sizes)
        ]
        # Use the minimum to ensure we don't run out of data
        return min(max_batches_per_dataset) if max_batches_per_dataset else 0
    
    def __iter__(self):
        """Generate batch indices - memory-efficient version using modulo indexing"""
        # Set random seed for reproducibility
        rng = np.random.RandomState(self.seed)
        
        # Generate shuffled indices for each dataset (only once, not repeated)
        indices_per_dataset = []
        for length in self.dataset_lengths:
            indices = np.arange(length)
            if self.shuffle:
                rng.shuffle(indices)
            indices_per_dataset.append(indices)
        
        # Track current position in each dataset's index array
        current_positions = [0] * len(self.dataset_lengths)
        
        # Generate batches
        for batch_idx in range(self.num_batches):
            batch_indices = []
            for dataset_idx, bs in enumerate(self.batch_sizes):
                dataset_length = self.dataset_lengths[dataset_idx]
                dataset_indices = indices_per_dataset[dataset_idx]
                current_pos = current_positions[dataset_idx]
                
                # Get indices for this batch, using modulo to cycle through dataset
                batch_dataset_indices = []
                for _ in range(bs):
                    # Use modulo to cycle through the dataset if needed
                    idx = dataset_indices[current_pos % dataset_length]
                    batch_dataset_indices.append(int(idx))
                    current_pos += 1
                
                # Update position for next batch
                current_positions[dataset_idx] = current_pos
                
                # Store as tuples (dataset_idx, sample_idx)
                batch_indices.extend([(dataset_idx, idx) for idx in batch_dataset_indices])
            
            yield batch_indices
    
    def __len__(self) -> int:
        return self.num_batches


class MixedDataset(Dataset):
    """
    Mixed dataset that combines multiple datasets with specified ratios.
    
    This is a clean implementation that:
    - Uses a custom sampler instead of nested DataLoaders
    - Supports configuration-driven dataset creation
    - Provides unified batch format across different datasets
    - Memory-efficient with proper cleanup support
    """
    
    def __init__(
        self,
        configs: List[DatasetConfig],
        ratios: List[float],
        seq_len: int = 16,
        transform: Optional[Callable] = None,
        mode: str = 'train',
        logger = None
    ):
        """
        Args:
            configs: List of dataset configurations
            ratios: Sampling ratios for each dataset
            seq_len: Sequence length
            transform: Transform to apply
            mode: 'train' or 'val'/'test'
        """
        assert len(configs) == len(ratios), f"Number of configs must match number of ratios: {len(configs)} != {len(ratios)}"
        assert all(r > 0 for r in ratios), f"All ratios must be positive: {ratios}"
        
        self.configs = configs
        self.ratios = ratios
        self.seq_len = seq_len
        self.transform = transform
        self.mode = mode if mode != 'test' else 'val'  # Normalize 'test' to 'val'
        
        # Initialize dataset registry (idempotent)
        DatasetRegistry.initialize_default_factories()
        
        # Create datasets and factories
        self.datasets: List[Dataset] = []
        self.factories: List[DatasetFactory] = []
        
        for config in configs:
            factory = DatasetRegistry.get_factory(config.name)
            self.factories.append(factory)
            
            if self.mode == 'train':
                dataset = factory.create_train_dataset(config, seq_len, transform)
            else:
                dataset = factory.create_val_dataset(config, seq_len, transform)
            
            self.datasets.append(dataset)
            logger.info(f"Loaded {config.name} {self.mode} dataset: {len(dataset)} samples")
        
        # Store dataset lengths
        self.dataset_lengths = [len(d) for d in self.datasets]
        
        # Create name to index mapping
        self.name_to_idx = {config.name: idx for idx, config in enumerate(configs)}
    
    def __len__(self) -> int:
        """
        Total length is sum of all dataset lengths.
        Note: When using with MixedBatchSampler, the actual number of samples
        will be determined by the sampler.
        """
        return sum(self.dataset_lengths)
    
    def __getitem__(self, idx: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """
        Get item by (dataset_idx, sample_idx) tuple.
        
        Args:
            idx: Tuple of (dataset_idx, sample_idx)
            
        Returns:
            Dictionary with 'observations' and 'dataset_idx' keys
        """
        dataset_idx, sample_idx = idx
        
        # Get batch from the specific dataset
        batch = self.datasets[dataset_idx][sample_idx]
        
        # Normalize batch using the appropriate factory
        observations = self.factories[dataset_idx].normalize_batch(batch)
        
        return {
            'observations': observations,
            'dataset_idx': dataset_idx
        }
    
    def clear_cache(self):
        """Clear any cached data in underlying datasets to free memory"""
        for dataset in self.datasets:
            if hasattr(dataset, 'clear_cache'):
                dataset.clear_cache()
            elif hasattr(dataset, 'json_cache'):
                # LoopNavDataset
                dataset.json_cache.clear()
                if hasattr(dataset, '_cache_order'):
                    dataset._cache_order.clear()
        import gc
        gc.collect()
    
    def __del__(self):
        """Cleanup when dataset is deleted"""
        try:
            self.clear_cache()
        except Exception:
            pass  # Ignore cleanup errors


def collate_mixed_batch(batch: List[Dict[str, torch.Tensor]], seq_len: Optional[int] = None) -> Dict[str, torch.Tensor]:
    """
    Collate function for mixed dataset batches with time-dimension alignment.
    
    Args:
        batch: List of samples from __getitem__
        seq_len: Desired time length; pad/truncate per sample to this length.
        
    Returns:
        Dictionary with batched observations and dataset indices
    """
    observations = []
    dataset_indices = []
    for sample in batch:
        obs = sample['observations']
        observations.append(obs)
        dataset_indices.append(sample['dataset_idx'])
    
    # Stack observations (B, T, C, H, W)
    try:
        observations = torch.stack(observations, dim=0)
    except Exception as e:
        # Fallback: keep list to avoid silent shape errors
        print(f"Warning: Cannot stack observations: {e}")
    
    dataset_indices = torch.tensor(dataset_indices, dtype=torch.long)
    
    return {
        'observations': observations,
        'dataset_indices': dataset_indices
    }


def create_mixed_dataloader(
    configs: List[DatasetConfig],
    ratios: List[float],
    batch_size: int,
    seq_len: int = 16,
    transform: Optional[Callable] = None,
    mode: str = 'train',
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42,
    logger = None
):
    """
    Convenient function to create a mixed dataset DataLoader.
    
    Args:
        configs: List of dataset configurations
        ratios: Sampling ratios for each dataset
        batch_size: Batch size
        seq_len: Sequence length
        transform: Transform to apply
        mode: 'train' or 'val'
        num_workers: Number of DataLoader workers
        pin_memory: Whether to pin memory
        shuffle: Whether to shuffle
        seed: Random seed
        
    Returns:
        DataLoader with mixed dataset
    """
    from torch.utils.data import DataLoader
    
    # Create mixed dataset
    dataset = MixedDataset(
        configs=configs,
        ratios=ratios,
        seq_len=seq_len,
        transform=transform,
        mode=mode,
        logger=logger
    )
    
    # Create sampler
    sampler = MixedBatchSampler(
        dataset_lengths=dataset.dataset_lengths,
        ratios=ratios,
        batch_size=batch_size,
        shuffle=shuffle,
        seed=seed,
        drop_last=True
    )
    
    # Handle collate function for ProcGen datasets
    # has_procgen = any(config.name in ['miner', 'maze'] for config in configs)
    # if has_procgen:
    #     from tensordict import TensorDict
    #     # Custom collate that handles both ProcGen TensorDict and regular tensors, with seq_len alignment
    #     def mixed_collate(batch):
    #         return collate_mixed_batch(batch, seq_len=seq_len)
    #     collate_fn = mixed_collate
    # else:
    collate_fn = lambda batch: collate_mixed_batch(batch, seq_len=seq_len)
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=(num_workers > 0)
    )
    
    return dataloader

