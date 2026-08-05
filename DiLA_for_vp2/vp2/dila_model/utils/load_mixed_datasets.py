"""
Utility functions to load mixed datasets from configuration files.

This module provides convenient functions to create mixed dataloaders
from YAML configuration files.
"""

import yaml
import torch
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.mixed_dataset import (
    DatasetConfig,
    create_mixed_dataloader,
    MixedDataset,
    MixedBatchSampler,
    PROCGEN_ENV_NAMES,
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_transform_from_config(config: Dict[str, Any], mode: str = 'train') -> transforms.Compose:
    """
    Create image transforms from configuration.
    
    Args:
        config: Configuration dictionary
        mode: 'train' or 'val'
    
    Returns:
        Composed transforms
    """
    transform_config = config.get('transforms', {})
    
    transform_list = []
    
    # Resize
    if 'resize' in transform_config:
        size = transform_config['resize']
        transform_list.append(transforms.Resize(tuple(size)))
    
    # Training augmentations
    if mode == 'train' and transform_config.get('train_augmentation', {}).get('random_crop', False):
        transform_list.append(transforms.RandomCrop(224))
    
    if mode == 'train' and transform_config.get('train_augmentation', {}).get('random_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if mode == 'train' and transform_config.get('train_augmentation', {}).get('color_jitter', False):
        transform_list.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1))
    
    # To tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalization
    if 'normalize' in transform_config:
        norm_config = transform_config['normalize']
        mean = norm_config.get('mean', [0.485, 0.456, 0.406])
        std = norm_config.get('std', [0.229, 0.224, 0.225])
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    return transforms.Compose(transform_list)


def create_dataset_configs_from_config(config: Dict[str, Any]) -> list:
    """
    Create DatasetConfig objects from configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        List of DatasetConfig objects
    """
    dataset_names = config['data']['dataset_names']
    configs = []
    
    for dataset_name in dataset_names:
        if dataset_name == "ssv2":
            ssv2_config = config['data']['ssv2']
            configs.append(
                DatasetConfig(
                    name="ssv2",
                    data_dir=ssv2_config['data_dir'],
                    train_file=ssv2_config['train_file'],
                    val_file=ssv2_config['val_file'],
                    additional_params={
                        'frames_per_clip': ssv2_config.get('frames_per_clip', 16),
                        'sample_downsample_rate': ssv2_config.get('sample_downsample_rate', 1)
                    }
                )
            )
        
        elif dataset_name == "procgen":
            procgen_config = config['data']['procgen']
            for game_name in procgen_config['dataset_names']:
                configs.append(
                    DatasetConfig(
                        name=game_name,
                        data_dir=procgen_config['data_dir'],
                        additional_params={
                            'split_ratio': procgen_config.get('split_ratio', 0.8),
                            'seed': procgen_config.get('seed', 42),
                            'return_goal': procgen_config.get('return_goal', True),
                            'cache_metadata': procgen_config.get('cache_metadata', False)
                        }
                    )            
                )       
                
        elif dataset_name == "openx":
            # Open-X-Embodiment datasets that have been pre-processed into folder + frames format
            openx_config = config['data']['openx']
            configs.append(
                DatasetConfig(
                    name="openx",
                    data_dir=openx_config['data_dir'],
                    additional_params={
                        'frame_skip': openx_config.get('frame_skip', 1),
                        'dataset_names': openx_config.get('dataset_names', None),
                        'max_episodes_per_dataset': openx_config.get('max_episodes_per_dataset', None),
                        'train_ratio': openx_config.get('train_ratio', 0.9),
                        'seed': openx_config.get('seed', 42),
                        'return_instruction': openx_config.get('return_instruction', False),
                        # Memory optimization: use lazy loading by default
                        'lazy_frame_list': openx_config.get('lazy_frame_list', True),
                    }
                )
            )
        
        elif dataset_name == "nwm":
            nwm_config = config['data']['nwm']
            configs.append(
                DatasetConfig(
                    name="nwm",
                    data_dir=nwm_config['data_dir'],
                    additional_params={
                        'data_split_folder': nwm_config.get('data_split_folder', nwm_config['data_dir']),
                        'dataset_name': nwm_config.get('dataset_name', 'recon'),
                        'image_size': nwm_config.get('image_size', (128, 128)),
                        'min_dist_cat': nwm_config.get('min_dist_cat', 3),
                        'max_dist_cat': nwm_config.get('max_dist_cat', 10),
                        'traj_names': nwm_config.get('traj_names', 'traj_names.txt'),
                        'normalize': nwm_config.get('normalize', True),
                        'predefined_index': nwm_config.get('predefined_index', None),
                    }
                )
            )
        
        elif dataset_name == "loopnav":
            loopnav_config = config['data']['loopnav']
            configs.append(
                DatasetConfig(
                    name="loopnav",
                    data_dir=loopnav_config['data_dir'],
                    additional_params={
                        'split_ratio': loopnav_config.get('split_ratio', 0.8),
                        'seed': loopnav_config.get('seed', 42),
                        'return_goal': loopnav_config.get('return_goal', True),
                        # Memory optimization: disable caching by default for large datasets
                        'cache_metadata': loopnav_config.get('cache_metadata', False),
                        'max_cached_meta': loopnav_config.get('max_cached_meta', 1000),
                    }
                )
            )
        
        elif dataset_name == "memorymaze":
            memorymaze_config = config['data']['memorymaze']
            configs.append(
                DatasetConfig(
                    name="memorymaze",
                    data_dir=memorymaze_config['data_dir'],
                    additional_params={
                        'split_ratio': memorymaze_config.get('split_ratio', 0.8),
                        'seed': memorymaze_config.get('seed', 42),
                        'return_goal': memorymaze_config.get('return_goal', False),
                        'return_state': memorymaze_config.get('return_state', False),
                        'cache_metadata': memorymaze_config.get('cache_metadata', True),
                    }
                )
            )
        
        elif dataset_name == "dmc_vision":
            dmc_config = config['data']['dmc_vision']
            configs.append(
                DatasetConfig(
                    name="dmc_vision",
                    data_dir=dmc_config['data_dir'],
                    additional_params={
                        'domain_name': dmc_config.get('domain_name', 'walker'),
                        'task_name': dmc_config.get('task_name', None),
                        'policy_level': dmc_config.get('policy_level', 'expert'),
                        'difficulty': dmc_config.get('difficulty', 'none'),
                        'dynamic_distractors': dmc_config.get('dynamic_distractors', False),
                        'img_height': dmc_config.get('img_height', 64),
                        'img_width': dmc_config.get('img_width', 64),
                        'obs_vrange': dmc_config.get('obs_vrange', [0.0, 1.0]),
                        'target_hidden': dmc_config.get('target_hidden', False),
                        'ant_fixed_seed': dmc_config.get('ant_fixed_seed', None),
                        'max_samples': dmc_config.get('max_samples', None),
                        'stride': dmc_config.get('stride', None),
                        'train_split': dmc_config.get('train_split', 'train'),
                        'val_split': dmc_config.get('val_split', 'test'),
                    }
                )
            )

        else:
            raise ValueError(f"Unsupported dataset name '{dataset_name}'. Known options: {dataset_names}")
    
    return configs


def get_ratios_for_epoch(config: Dict[str, Any], epoch: int) -> list:
    """
    Get mixing ratios for a specific epoch (supports curriculum learning).
    
    Args:
        config: Configuration dictionary
        epoch: Current epoch number
    
    Returns:
        List of mixing ratios
    """
    if not config.get('training', {}).get('use_curriculum', False):
        return config['data']['ratios']
    
    # Curriculum learning
    curriculum_stages = config['training']['curriculum_stages']
    
    for stage in curriculum_stages:
        epoch_range = stage['epochs']
        if epoch_range[0] <= epoch <= epoch_range[1]:
            return stage['ratios']
    
    # Default to last stage ratios if epoch is beyond curriculum
    return curriculum_stages[-1]['ratios']


def create_dataloaders_from_config(
    config_path: str,
    epoch: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders from a config file.
    
    Args:
        config_path: Path to YAML configuration file
        epoch: Current epoch (for curriculum learning)
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Load config
    config = load_config(config_path)
    
    # Get parameters
    data_config = config['data']
    seq_len = data_config['seq_len']
    batch_size = data_config['batch_size']
    num_workers = data_config.get('num_workers', 4)
    pin_memory = data_config.get('pin_memory', True)
    shuffle_train = data_config.get('shuffle_train', True)
    seed = data_config.get('seed', 42)
    
    # Create dataset configs
    dataset_configs = create_dataset_configs_from_config(config)
    
    # Get ratios (with curriculum support)
    ratios = get_ratios_for_epoch(config, epoch)
    
    # Create transforms
    train_transform = create_transform_from_config(config, mode='train')
    val_transform = create_transform_from_config(config, mode='val')
    
    # Create dataloaders
    train_loader = create_mixed_dataloader(
        configs=dataset_configs,
        ratios=ratios,
        batch_size=batch_size,
        seq_len=seq_len,
        transform=train_transform,
        mode='train',
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle_train,
        seed=seed + epoch  # Different seed per epoch
    )
    
    val_config = config.get('validation', {})
    val_batch_size = val_config.get('val_batch_size', batch_size)
    
    val_loader = create_mixed_dataloader(
        configs=dataset_configs,
        ratios=ratios,
        batch_size=val_batch_size,
        seq_len=seq_len,
        transform=val_transform,
        mode='val',
        num_workers=max(1, num_workers // 2),  # Use fewer workers for validation
        pin_memory=pin_memory,
        shuffle=False,
        seed=seed
    )
    
    return train_loader, val_loader


def print_dataloader_info(train_loader: DataLoader, val_loader: DataLoader, config: Dict[str, Any], epoch: int = 0):
    """Print information about the dataloaders"""
    print("=" * 80)
    print("Mixed Dataset Loader Configuration")
    print("=" * 80)
    
    dataset_names = config['data']['dataset_names']
    ratios = get_ratios_for_epoch(config, epoch)
    batch_size = config['data']['batch_size']
    
    print(f"\nDatasets: {', '.join(dataset_names)}")
    print(f"Mixing ratios: {ratios}")
    print(f"Batch size: {batch_size}")
    
    # Calculate samples per dataset in a batch
    total_ratio = sum(ratios)
    print("\nSamples per batch from each dataset:")
    for name, ratio in zip(dataset_names, ratios):
        samples = int(batch_size * ratio / total_ratio)
        print(f"  {name}: ~{samples} samples")
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print("=" * 80)


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load mixed datasets from config")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--epoch", type=int, default=0, help="Epoch number (for curriculum learning)")
    parser.add_argument("--test-load", action="store_true", help="Test loading a batch")
    
    args = parser.parse_args()
    
    # Load dataloaders
    print(f"Loading dataloaders from config: {args.config}")
    config = load_config(args.config)
    train_loader, val_loader = create_dataloaders_from_config(args.config, epoch=args.epoch)
    
    # Print info
    print_dataloader_info(train_loader, val_loader, config, epoch=args.epoch)
    
    # Test loading a batch
    if args.test_load:
        print("\nTesting batch loading...")
        for batch in train_loader:
            observations = batch['observations']
            dataset_indices = batch['dataset_indices']
            print(f"Batch shape: {observations.shape}")
            print(f"Dataset indices: {dataset_indices}")
            print(f"Unique datasets in batch: {torch.unique(dataset_indices).tolist()}")
            break
        print("Successfully loaded a batch!")

