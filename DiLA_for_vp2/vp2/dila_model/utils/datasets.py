# -*- coding: utf-8 -*-
import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import json
import einops
from pathlib import Path
from torch import stack
import torchvision.transforms.functional as F
import re
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from typing import Optional, Callable, List, Dict, Any, Tuple, Union
import pickle
import tqdm


def normalize_01_into_pm1(x):  # normalize x from [0, 1] to [-1, 1] by (x*2) - 1
    return x.add(x).add_(-1)


class Adapted_Transform:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, values):
        obs, act, mask = values
        t, v, c, h, w = obs.shape
        obs = obs.view(-1, c, h, w)
        transformed_images = stack([self.transform(F.to_pil_image(img)) for img in obs])
        _, c, h, w  = transformed_images.shape
        transformed_images = transformed_images.view(t, v, c, h, w)
        return transformed_images, act, mask


def transpose_batch_timestep(*args):
    return (einops.rearrange(arg, "b t ... -> t b ...") for arg in args)


class OmniDataset(Dataset):
    def __init__(self, img_dir, seq_len=20, action_list=np.arange(-30, 30 + 1, 5), transform=None, continuous_rotation=False, initial_rotation_range=np.arange(0, 360, 5), batch_initial_rotation=None):
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.continuous_rotation = continuous_rotation

        # Collect image file information
        self.all_object_dir = self.count_lowest_level_directories(self.img_dir)
        self.indices = np.arange(len(self.all_object_dir))
        self.initial_rotation_range = initial_rotation_range # default 72 rotations
        self.action_list = np.array([5]) if self.continuous_rotation else action_list
        self.batch_initial_rotation = batch_initial_rotation
        
        # Record category
        category_name_lst = []
        for path in self.all_object_dir:
            category_name, object_name = self.extract_category_from_path(path)
            category_name_lst.append(category_name)
        category_name_lst = set(category_name_lst)
        self.category_index_dict = {category: idx for idx, category in enumerate(sorted(category_name_lst))}
        self.category_num = len(category_name_lst)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        initial_rotation = self.batch_initial_rotation[idx] if self.batch_initial_rotation is not None else random.choice(self.initial_rotation_range)
        action_seq = random.choices(self.action_list.tolist(), k=self.seq_len - 1)
        relative_rotation_seq = np.cumsum(action_seq)
        relative_rotation_seq = np.insert(relative_rotation_seq, 0, 0)
        rotation_seq = (relative_rotation_seq + initial_rotation) % 360
        image_seq = []
        for rotation in rotation_seq:
            image = self.load_image(idx, rotation)
            image_seq.append(image)
        # Add a zero action at the end of the sequence
        # action_seq = np.pad(action_seq, (0, 1), 'constant', constant_values=0)
        category_name, object_name = self.extract_category_from_path(self.all_object_dir[idx])
        category = [self.category_index_dict[category_name]]*self.seq_len

        item ={     
                'image_seq': torch.stack(image_seq).float(),                            # (seq_len, 3, 128, 128)
                'rotation_seq': torch.tensor(rotation_seq).float(),                     # (seq_len)
                'action_seq': torch.tensor(action_seq).float(),                         # (seq_len-1)
                'relative_rotation_seq': torch.tensor(relative_rotation_seq).float(),   # (seq_len)
                'seq_index': torch.arange(0, self.seq_len, step=1).float(),                  # (seq_len)
                'category_name': category_name,  # (1) 
                'object_name': object_name,  # (1)  
                'category':  torch.tensor(category).long(),                       # (seq_len)          
        }
        return item  # (seq_len, 3, 128, 128), (seq_len), (seq_len(pad 0 at the end))
    
    def count_lowest_level_directories(self, path):
        lowest_level_dirs = []
        for root, dirs, files in os.walk(path):
            if not dirs: # if the directory is the lowest level
                    lowest_level_dirs.append(root)
        return lowest_level_dirs
    
    def load_image(self, idx, rotation):
        """Load and preprocess a single image."""
        img_path = self.all_object_dir[idx] + f'/{rotation//5:03d}.png'
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(), 
            ])
            image = transform(image)
        return image

    def extract_category_from_path(self, path):
        parent_dir = os.path.basename(os.path.dirname(path))
        match = re.match(r"^(\D+)", parent_dir)
        if match:
            return match.group(1).rstrip('_'), parent_dir 
        return "", parent_dir


class COIL100_DataLoader(Dataset):
    def __init__(self, img_dir, batch_size, seq_len=10, action_threshold=30, batch_cls=4, shuffle=False, transform=None):
        # Initialize parameters
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_cls = batch_cls
        self.shuffle = shuffle
        self.transform = transform

        # Collect image file information
        self.labels_list = [f for f in os.listdir(img_dir) if f.endswith('.png')]
        self.indices = np.arange(1, 101)  # 100 classes
        self.rotation_list = np.arange(0, 360, 5)  # 72 rotations
        self.action_list = np.arange(-action_threshold, action_threshold + 1, 5)

        if self.shuffle:
            np.random.shuffle(self.indices)

    def __iter__(self):
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self

    def __next__(self):
        # Check end of dataset
        if self.current_index >= len(self.indices):
            raise StopIteration

        # Get current batch class indices
        start = self.current_index
        end = min(start + self.batch_cls, len(self.indices))
        self.current_index = end

        cls_indices = self.indices[start:end]
        cls_seq_num = self.batch_size // len(cls_indices)  # Sequences per class

        # Load image sequences
        X = []
        X_rotation = []
        action_seq_list = []
        for cls in cls_indices:
            # Generate rotation and action sequences
            initial_rotation = self.efficient_sample_with_min_distance(self.rotation_list, cls_seq_num, len(self.rotation_list) // cls_seq_num)
            rotation_seq_samples, action_seq_samples = [], []

            for i in range(cls_seq_num):
                action_seq = random.choices(self.action_list.tolist(), k=self.seq_len - 1)
                rotation_seq = np.cumsum(action_seq) + initial_rotation[i]
                rotation_seq = np.insert(rotation_seq, 0, initial_rotation[i]) % 360
                rotation_seq_samples.append(rotation_seq)
                action_seq_samples.append(action_seq)

            rotation_seq_samples = np.array(rotation_seq_samples)
            action_seq_samples = np.array(action_seq_samples)
            X_rotation.append(rotation_seq_samples)
            action_seq_list.append(action_seq_samples)
            for rotation_seq in rotation_seq_samples:
                image_seq = [self._load_image(cls, rotation) for rotation in rotation_seq]
                X.append(image_seq)

        # Create labels
        label = np.repeat(cls_indices[:, np.newaxis] - 1, cls_seq_num * self.seq_len).reshape(-1, self.seq_len)
        
        # Create rotation and action data
        X_rotation = np.stack(X_rotation).reshape(-1, self.seq_len)
        action_seq_list = np.stack(action_seq_list).reshape(-1, self.seq_len-1)

        return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(X_rotation, dtype=torch.float32), \
            torch.tensor(action_seq_list, dtype=torch.float32), torch.tensor(label, dtype=torch.long), torch.tensor(cls_seq_num, dtype=torch.long)

    def _load_image(self, cls, rotation):
        """Load and preprocess a single image."""
        img_path = os.path.join(self.img_dir, f'obj{cls}__{rotation}.png')
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image).numpy()
        return image

    def __len__(self):
        return (len(self.indices) + self.batch_cls - 1) // self.batch_cls

    def get_sequence(self, seq_num):
        X = []
        np.random.shuffle(self.indices)
        cls_indices = self.indices[0:seq_num]
        for cls in cls_indices:
            image_seq = [self._load_image(cls, rotation) for rotation in self.rotation_list]
            X.append(image_seq)

        X_rotation = np.tile(self.rotation_list, (len(cls_indices), 1))
        action_seq_list = np.tile(np.ones(len(self.rotation_list)-1) * 5., (len(cls_indices), 1))
        label = np.zeros(len(self.rotation_list)) + cls

        return torch.tensor(np.array(X)), torch.tensor(X_rotation), torch.tensor(action_seq_list), torch.tensor(label)
    
    def efficient_sample_with_min_distance(self, array, num_points, min_distance):
        total_points = len(array)
        if num_points * min_distance > total_points:
            raise ValueError("Please choose a smaller num_points or larger min_distance.")
        
        start_idx = np.random.randint(0, total_points)
        
        sampled_indices = [(start_idx + i * min_distance) % total_points for i in range(num_points)]
        
        sampled_points = [array[idx] for idx in sampled_indices]
        return sampled_points
    

class MIRO_Dataloader(Dataset):
    def __init__(self, img_dir, batch_size, seq_len=10, action_threshold=30, batch_cls=4, shuffle=False, transform=None):
        # Initialize parameters
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.batch_cls = batch_cls
        self.shuffle = shuffle
        self.transform = transform
        self.seq_len = 16

        self.cls_type = ["bus", "car", "cleanser", "clock", "cup", "headphones", "mouse", "scissors", "shoe", "stapler", "sunglasses", "tape_cutter"]

        if self.shuffle:
            np.random.shuffle(self.indices)

    def _load_image(self, root_dir, image_files, idx):
        """Load and preprocess a single image."""
        img_path = os.path.join(root_dir, image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image).numpy()
        return image

    def get_sequence(self):
        X = []

        for cls in self.cls_type:
            img_path = os.path.join(self.img_dir, f'{cls}')
            image_files = sorted([f for f in os.listdir(img_path) if f.endswith('.png')])
            image_seq = [self._load_image(img_path, image_files, i) for i in range(self.seq_len)]
            X.append(image_seq)

        X_rotation = np.tile(np.arange(0, 360, 22.5), (len(self.cls_type), 1))
        action_seq_list = np.tile(np.ones(self.seq_len-1) * 22.5, (len(self.cls_type), 1))

        return torch.tensor(np.array(X)), torch.tensor(X_rotation), torch.tensor(action_seq_list)

class SomethingSomethingV2Dataset(Dataset):
    def __init__(self, root_dir, annotations_file, transform=None, frames_per_clip=8, step_between_clips=None, sample_downsample_rate=10, sliding_window=4):
        """
        Args:
            root_dir (str): Directory with all the images.
            annotations_file (str): Path to the annotations file (JSON format).
            transform (callable, optional): Optional transform to be applied on a sample.
            frames_per_clip (int): Number of frames per clip.
            step_between_clips (int): Step in frames between each clip. If None, it will be set to frames_per_clip to minimize overlap.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.frames_per_clip = frames_per_clip
        self.step_between_clips = step_between_clips if step_between_clips is not None else frames_per_clip
        self.sliding_window = sliding_window

        with open(annotations_file, 'r') as f:
            samples = [sample for sample in json.load(f) if len(os.listdir(os.path.join(root_dir, sample['id']))) >= frames_per_clip]

        # Downsample the number of samples
        self.samples = samples[::sample_downsample_rate]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_id = sample['id']

        # Get list of all frame files in the video directory
        video_dir = os.path.join(self.root_dir, video_id)
        frame_files = sorted([f for f in os.listdir(video_dir) if f.endswith('.jpg')])
        num_frames = len(frame_files)

        clip = []
        for frame_idx in np.linspace(0, num_frames, self.frames_per_clip, dtype=int, endpoint=False):
            frame_path = os.path.join(video_dir, frame_files[frame_idx])
            image = Image.open(frame_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            clip.append(image)

        clip = torch.stack(clip, dim=0)  # [4, C, H, W]
        # Create sliding window clips
        if self.sliding_window > 1:
            # Ensure the clip is long enough for the sliding window
            if len(clip) < self.sliding_window:
                raise ValueError(f"Clip length {len(clip)} is less than sliding window size {self.sliding_window}.")
            
            # Create sliding window clips
            # clips = [clip[i:i+self.sliding_window] for i in range(self.frames_per_clip - self.sliding_window + 1)]
            if self.frames_per_clip == self.sliding_window:
                clips = [clip]
            else: 
                # clips = [clip[0:self.sliding_window], clip[self.frames_per_clip-self.sliding_window:self.frames_per_clip]]
                # clips = [clip[i:i+self.sliding_window] for i in range(self.frames_per_clip - self.sliding_window + 1)]
                clips = [clip[4:4+self.sliding_window]]

        return torch.stack(clips, dim=0) # [5, 4, C, H, W], video_id



class MultiActionOmniDataset(Dataset):
    def __init__(self, img_dir, seq_len=20, 
                 rotation_actions=np.arange(-30, 30 + 1, 5),
                 plane_rotation_actions=np.arange(-30, 30 + 1, 5),
                 scale_actions=np.arange(0.5, 2 + 0.1, 0.1),
                 translation_x_actions=np.arange(-20, 20 + 1, 5),
                 translation_y_actions=np.arange(-20, 20 + 1, 5),
                 transform=None, continuous_rotation=False,  
                 initial_rotation_range=np.arange(0, 360, 5),
                 initial_plane_rotation_range=np.arange(-45, 45 + 1, 15),
                 initial_scale_range=np.arange(0.5, 1.0 + 0.01, 0.1),
                 initial_translation_x_range=np.arange(-30, 30 + 1, 10),
                 initial_translation_y_range=np.arange(-30, 30 + 1, 10),
                 action_types=['rotation', 'plane_rotation', 'scale', 'translation_x', 'translation_y'],
                 single_action_per_step=False,
                 background_path=None,
                 use_mask=True,
                 image_size=(128, 128)):
        """
        Multi-action dataset with improved initialization and boundary constraints.
        
        Args:
            img_dir (str): Directory containing object images.
            seq_len (int): Length of the sequence.
            rotation_actions (np.ndarray): Actions for rotation.
            scale_actions (np.ndarray): Actions for scaling.
            translation_x_actions (np.ndarray): Actions for x translation.
            translation_y_actions (np.ndarray): Actions for y translation.
            plane_rotation_actions (np.ndarray): Actions for plane rotation.
            transform (callable, optional): Transform to apply to images.
            continuous_rotation (bool): Whether to allow continuous rotation.
            initial_rotation_range (np.ndarray): Range of initial rotations.
            initial_scale_range (np.ndarray): Range of initial scales.
            initial_translation_x_range (np.ndarray): Range of initial x translations.
            initial_translation_y_range (np.ndarray): Range of initial y translations.
            initial_plane_rotation_range (np.ndarray): Range of initial plane rotations.
            action_types (list): Types of actions to include in the dataset.
            background_path (str, optional): Path to background image(s).
            use_mask (bool): Whether to use masks with images.
            image_size (tuple): Size of the output images.

            # Example additional transforms:
            additional_transforms = transforms.Compose([
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
                # Do not add ToTensor() here; tensor conversion is handled below.
            ])
        """
        self.img_dir = img_dir
        self.seq_len = seq_len
        self.transform = transform
        self.continuous_rotation = continuous_rotation
        self.action_types = action_types
        self.use_mask = use_mask
        self.background_path = background_path
        self.image_size = image_size

        # Load background image(s)
        self.background = self._load_background()

        # Collect image file information
        self.all_object_dir = self.count_lowest_level_directories(self.img_dir)
        self.indices = np.arange(len(self.all_object_dir))
        
        # Validate the initial scale range.
        if np.min(initial_scale_range) < 0.3:
            print(f"Warning: initial_scale_range minimum {np.min(initial_scale_range)} < 0.3")

        # Initial ranges
        self.initial_rotation_range = initial_rotation_range
        self.initial_plane_rotation_range = initial_plane_rotation_range
        self.initial_scale_range = initial_scale_range
        self.initial_translation_x_range = initial_translation_x_range
        self.initial_translation_y_range = initial_translation_y_range
        
        # Action ranges
        self.rotation_actions = np.array([5]) if self.continuous_rotation else rotation_actions
        self.plane_rotation_actions = plane_rotation_actions
        self.scale_actions = scale_actions
        self.translation_x_actions = translation_x_actions
        self.translation_y_actions = translation_y_actions
        
        # Treat the original image as the maximum scale.
        self.max_scale = 1.0
        
        # Estimate the approximate object size.
        self.estimated_object_size = min(self.image_size) * 0.8
        
        # Record category
        category_name_lst = []
        for path in self.all_object_dir:
            category_name, object_name = self.extract_category_from_path(path)
            category_name_lst.append(category_name)
        category_name_lst = set(category_name_lst)
        self.category_index_dict = {category: idx for idx, category in enumerate(sorted(category_name_lst))}
        self.category_num = len(category_name_lst)
        
        self.single_action_per_step = single_action_per_step

    def _load_background(self):
        """Load background image(s)."""
        if self.background_path is None:
            # Use gray background
            return None
        elif os.path.isfile(self.background_path):
            # Single background image
            bg = Image.open(self.background_path).convert('RGB')
            return bg
        elif os.path.isdir(self.background_path):
            # Multiple background images (commented out for future use)
            # bg_files = [f for f in os.listdir(self.background_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            # backgrounds = [Image.open(os.path.join(self.background_path, f)).convert('RGB') for f in bg_files]
            # return backgrounds
            
            # For now, use the first background image
            bg_files = [f for f in os.listdir(self.background_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            if bg_files:
                bg = Image.open(os.path.join(self.background_path, bg_files[0])).convert('RGB')
                return bg
            else:
                return None
        else:
            return None

    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, key):
        # Slice access returns a batch accessor: dataset[:].
        if isinstance(key, slice):
            return BatchAccessor(self, key)
        
        # List/array access also returns a batch accessor: dataset[[0, 1, 2]].
        elif isinstance(key, (list, tuple, np.ndarray)):
            return BatchAccessor(self, key)
        
        # Single-index access returns one item.
        else:
            return self._get_single_item(key)

    def _get_single_item(self, idx):
        # Randomly initialize all transform parameters.
        initial_rotation = random.choice(self.initial_rotation_range)
        initial_plane_rotation = random.choice(self.initial_plane_rotation_range)
        initial_scale = random.choice(self.initial_scale_range)
        initial_translation_x = random.choice(self.initial_translation_x_range)
        initial_translation_y = random.choice(self.initial_translation_y_range)
        
        # Keep the initial object position within image bounds.
        initial_translation_x, initial_translation_y = self._clamp_translation(
            initial_translation_x, initial_translation_y, initial_scale
        )
        
        # Generate actions from the initial state.
        action_seq = self._generate_action_sequence(
            initial_plane_rotation, initial_scale, initial_translation_x, initial_translation_y
        )
        
        # Compute the full transform sequence.
        transform_seq = self._calculate_transform_sequence_with_bounds(
            action_seq, initial_rotation, initial_plane_rotation, initial_scale, 
            initial_translation_x, initial_translation_y
        )
        
        # Load and transform images
        image_seq = []
        for i, transform_params in enumerate(transform_seq):
            image = self.load_and_transform_image(idx, transform_params)
            image_seq.append(image)
        
        # Get category information
        category_name, object_name = self.extract_category_from_path(self.all_object_dir[idx])
        category = [self.category_index_dict[category_name]] * self.seq_len

        item = {
            'image_seq': torch.stack(image_seq).float(),
            'action_seq': torch.tensor(action_seq).float(),
            'transform_seq': torch.tensor(transform_seq).float(),
            'seq_index': torch.arange(0, self.seq_len, step=1).float(),
            'category_name': category_name,
            'object_name': object_name,
            'category': torch.tensor(category).long(),
        }
        return item

    def _clamp_translation(self, tx, ty, scale):
        """
        Clamp translation based on scale so the object stays within image bounds.
        """
        # Object size after scaling.
        scaled_object_size = self.estimated_object_size * scale
        
        # Maximum translation that keeps object edges inside the image.
        max_tx = (self.image_size[0] - scaled_object_size) / 2
        max_ty = (self.image_size[1] - scaled_object_size) / 2
        
        # Clamp translation.
        tx = np.clip(tx, -max_tx, max_tx)
        ty = np.clip(ty, -max_ty, max_ty)
        
        return tx, ty

    def _get_valid_translation_actions(self, current_tx, current_ty, current_scale):
        """
        Return translation actions that remain valid under the current scale.
        """
        valid_tx_actions = []
        valid_ty_actions = []
        
        for tx_action in self.translation_x_actions:
            new_tx = current_tx + tx_action
            new_tx_clamped, _ = self._clamp_translation(new_tx, current_ty, current_scale)
            if abs(new_tx_clamped - new_tx) < 1e-6:
                valid_tx_actions.append(tx_action)
        
        for ty_action in self.translation_y_actions:
            new_ty = current_ty + ty_action
            _, new_ty_clamped = self._clamp_translation(current_tx, new_ty, current_scale)
            if abs(new_ty_clamped - new_ty) < 1e-6:
                valid_ty_actions.append(ty_action)
        
        # If no valid action exists, keep the no-op action.
        if not valid_tx_actions:
            valid_tx_actions = [0]
        if not valid_ty_actions:
            valid_ty_actions = [0]
            
        return valid_tx_actions, valid_ty_actions

    def _generate_action_sequence(self, initial_plane_rotation, initial_scale, initial_tx, initial_ty):
        """
        Generate an action sequence, optionally activating one action per step.
        """
        action_seq = []
        action_dim = len(self.action_types)
        
        # Start from the provided initial state.
        current_plane_rotation = initial_plane_rotation
        current_scale = initial_scale
        current_tx = initial_tx
        current_ty = initial_ty
        
        for step in range(self.seq_len - 1):
            action = np.zeros(action_dim)
            
            if self.single_action_per_step:
                action = np.ones(action_dim)
                for i, action_type in enumerate(self.action_types):
                    if action_type != 'scale':
                        action[i] = 0.0
                # Activate exactly one action type at each step.
                active_action_idx = random.choice(range(len(self.action_types)))
                active_action_type = self.action_types[active_action_idx]
                
                # Set a nonzero value only for the selected action type.
                if active_action_type == 'rotation':
                    action[active_action_idx] = random.choice(self.rotation_actions)
                elif active_action_type == 'plane_rotation':
                    action[active_action_idx] = random.choice(self.plane_rotation_actions)
                elif active_action_type == 'scale':
                    valid_scale_action = self._sample_valid_scale_action(current_scale, step)
                    action[active_action_idx] = valid_scale_action
                elif active_action_type == 'translation_x':
                    valid_tx_actions, _ = self._get_valid_translation_actions(
                        current_tx, current_ty, current_scale
                    )
                    action[active_action_idx] = random.choice(valid_tx_actions)
                elif active_action_type == 'translation_y':
                    _, valid_ty_actions = self._get_valid_translation_actions(
                        current_tx, current_ty, current_scale
                    )
                    action[active_action_idx] = random.choice(valid_ty_actions)
                
            else:
                # All action types may be active in the same step.
                for i, action_type in enumerate(self.action_types):
                    if action_type == 'rotation':
                        action[i] = random.choice(self.rotation_actions)
                    elif action_type == 'plane_rotation':
                        action[i] = random.choice(self.plane_rotation_actions)
                    elif action_type == 'scale':
                        valid_scale_action = self._sample_valid_scale_action(current_scale, step)
                        action[i] = valid_scale_action
                    elif action_type == 'translation_x':
                        valid_tx_actions, _ = self._get_valid_translation_actions(
                            current_tx, current_ty, current_scale
                        )
                        action[i] = random.choice(valid_tx_actions)
                    elif action_type == 'translation_y':
                        _, valid_ty_actions = self._get_valid_translation_actions(
                            current_tx, current_ty, current_scale
                        )
                        action[i] = random.choice(valid_ty_actions)
            
            action_seq.append(action)
            
            # Update the running state.
            for i, action_type in enumerate(self.action_types):
                if action_type == 'plane_rotation':
                    current_plane_rotation += action[i]
                    current_plane_rotation = current_plane_rotation % 360
                elif action_type == 'scale':
                    if action[i] != 0:
                        new_scale = current_scale * action[i]
                        if not (0.3 - 1e-6 <= new_scale <= self.max_scale + 1e-6):
                            print(f"ERROR in action generation: step {step}, current_scale {current_scale}, action {action[i]}, new_scale {new_scale}")
                        current_scale = new_scale
                elif action_type == 'translation_x':
                    current_tx += action[i]
                elif action_type == 'translation_y':
                    current_ty += action[i]
            
            # Keep translation inside bounds after the update.
            current_tx, current_ty = self._clamp_translation(current_tx, current_ty, current_scale)
        
        return np.array(action_seq)
    
    def _sample_valid_scale_action(self, current_scale, step_info=None, max_attempts=50):
        """
        Sample a valid scale action.
        """
        min_scale = 0.3
        max_scale = self.max_scale
        
        # Precompute the valid action range.
        max_allowed_action = max_scale / current_scale
        min_allowed_action = min_scale / current_scale
        
        # Find all actions that are valid under the current scale.
        valid_actions = []
        for action in self.scale_actions:
            new_scale = current_scale * action
            if min_scale - 1e-6 <= new_scale <= max_scale + 1e-6:
                valid_actions.append(action)
        
        if not valid_actions:
            # Fall back to a safe no-op if no valid scale action exists.
            print(f"No valid scale actions for current_scale {current_scale}, step {step_info}")
            print(f"Scale actions: {self.scale_actions}")
            print(f"Required range: [{min_allowed_action:.3f}, {max_allowed_action:.3f}]")
            
            return 1.0
        
        # Sample from valid actions.
        return random.choice(valid_actions)

    def _calculate_transform_sequence_with_bounds(self, action_seq, initial_rotation, initial_plane_rotation, initial_scale, initial_tx, initial_ty):
        """Compute the transform sequence, including in-plane rotation."""
        transform_seq = []
        
        # Current state: rotation, plane rotation, scale, tx, ty.
        current_rotation = initial_rotation
        current_plane_rotation = initial_plane_rotation
        current_scale = initial_scale
        current_tx = initial_tx
        current_ty = initial_ty
        
        # Add the initial 5D transform state.
        transform_seq.append([current_rotation, current_plane_rotation, current_scale, current_tx, current_ty])
        
        # Apply actions cumulatively
        for step, action in enumerate(action_seq):
            for i, action_type in enumerate(self.action_types):
                if action_type == 'rotation':
                    current_rotation = (current_rotation + action[i]) % 360
                elif action_type == 'plane_rotation':
                    current_plane_rotation = (current_plane_rotation + action[i]) % 360
                elif action_type == 'scale':
                    old_scale = current_scale
                    current_scale *= action[i]
                    
                    if current_scale < 0.3 - 1e-6 or current_scale > self.max_scale + 1e-6:
                        print(f"ERROR: Invalid scale at step {step}")
                        print(f"  old_scale: {old_scale}")
                        print(f"  action: {action[i]}")
                        print(f"  new_scale: {current_scale}")
                        raise ValueError(f"Scale {current_scale} out of bounds!")
                        
                elif action_type == 'translation_x':
                    current_tx += action[i]
                elif action_type == 'translation_y':
                    current_ty += action[i]
            
            # Apply boundary constraints.
            current_tx, current_ty = self._clamp_translation(current_tx, current_ty, current_scale)
            
            transform_seq.append([current_rotation, current_plane_rotation, current_scale, current_tx, current_ty])
        
        return np.array(transform_seq)

    def load_and_transform_image(self, idx, transform_params):
        """Load image with mask and apply transformations: plane rotation -> scale -> translation."""
        rotation, plane_rotation, scale, tx, ty = transform_params
        
        # Load base image and mask at the specified rotation
        img_path = self.all_object_dir[idx] + f'/{int(rotation)//5:03d}.png'
        mask_path = self.all_object_dir[idx] + f'/{int(rotation)//5:03d}_mask.npy'
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        original_size = image.size
        
        if self.use_mask and os.path.exists(mask_path):
            # Load mask
            mask = np.load(mask_path)
            mask = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
            
            # 1. Apply plane rotation FIRST
            if plane_rotation != 0:
                image = image.rotate(plane_rotation, expand=False, fillcolor=(0, 0, 0, 0))
                mask = mask.rotate(plane_rotation, expand=False, fillcolor=0)
            
            # 2. Apply scaling SECOND
            if scale != 1.0:
                new_height = int(original_size[1] * scale)
                new_width = int(original_size[0] * scale)
                
                # Resize both image and mask
                image = image.resize((new_width, new_height), Image.LANCZOS)
                mask = mask.resize((new_width, new_height), Image.LANCZOS)
                
                # Create new canvas with original size
                scaled_image = Image.new('RGBA', original_size, (0, 0, 0, 0))
                scaled_mask = Image.new('L', original_size, 0)
                
                # Calculate position to center the scaled object
                paste_x = (original_size[0] - new_width) // 2
                paste_y = (original_size[1] - new_height) // 2
                
                # Paste scaled image and mask at center
                scaled_image.paste(image, (paste_x, paste_y))
                scaled_mask.paste(mask, (paste_x, paste_y))
                
                image = scaled_image.convert('RGB')
                mask = scaled_mask
            
            # 3. Apply translation LAST (in global screen coordinates)
            if tx != 0 or ty != 0:
                # Create new canvas for translation
                translated_image = Image.new('RGBA', original_size, (0, 0, 0, 0))
                translated_mask = Image.new('L', original_size, 0)
                
                # Calculate translation bounds
                src_x = max(0, -int(tx))
                src_y = max(0, -int(ty))
                dst_x = max(0, int(tx))
                dst_y = max(0, int(ty))
                
                width = original_size[0] - abs(int(tx))
                height = original_size[1] - abs(int(ty))
                
                if width > 0 and height > 0:
                    # Crop and paste the translated part
                    cropped_image = image.crop((src_x, src_y, src_x + width, src_y + height))
                    cropped_mask = mask.crop((src_x, src_y, src_x + width, src_y + height))
                    
                    translated_image.paste(cropped_image, (dst_x, dst_y))
                    translated_mask.paste(cropped_mask, (dst_x, dst_y))
                
                image = translated_image.convert('RGB')
                mask = translated_mask
            
            # Composite with background
            if self.background is not None:
                background = self.background.resize(original_size)
            else:
                background = Image.new('RGB', original_size, (128, 128, 128))
            
            final_image = Image.composite(image, background, mask)
            
        else:
            # If not using mask, apply transforms directly
            final_image = image
            
            # Apply same order: plane rotation -> scale -> translation
            if plane_rotation != 0:
                final_image = final_image.rotate(plane_rotation, expand=False, fillcolor=(128, 128, 128))
        
        # Apply user-defined transforms if provided (on PIL image first)
        if self.transform:
            if isinstance(self.transform, transforms.Compose):
                # Split PIL transforms from tensor transforms.
                pil_transforms = []
                tensor_transforms = []
                to_tensor_found = False
                
                for t in self.transform.transforms:
                    if isinstance(t, transforms.ToTensor):
                        to_tensor_found = True
                    elif not to_tensor_found:
                        # Transforms before ToTensor operate on PIL images.
                        pil_transforms.append(t)
                    else:
                        # Transforms after ToTensor operate on tensors.
                        tensor_transforms.append(t)
                
                # Apply PIL transforms.
                for t in pil_transforms:
                    final_image = t(final_image)
                
                # Convert to tensor.
                image_tensor = transforms.ToTensor()(final_image)
                
                # Apply tensor transforms.
                for t in tensor_transforms:
                    image_tensor = t(image_tensor)
                    
            else:
                # Single transform case.
                if isinstance(self.transform, transforms.ToTensor):
                    image_tensor = self.transform(final_image)
                else:
                    # Assume a PIL transform, then convert to tensor.
                    final_image = self.transform(final_image)
                    image_tensor = transforms.ToTensor()(final_image)
        else:
            # No additional transforms; convert directly to tensor.
            image_tensor = transforms.ToTensor()(final_image)
        
        return image_tensor
    def count_lowest_level_directories(self, path):
        """Same as OmniDataset."""
        lowest_level_dirs = []
        for root, dirs, files in os.walk(path):
            if not dirs:
                lowest_level_dirs.append(root)
        return lowest_level_dirs

    def extract_category_from_path(self, path):
        """Same as OmniDataset."""
        parent_dir = os.path.basename(os.path.dirname(path))
        match = re.match(r"^(\D+)", parent_dir)
        if match:
            return match.group(1).rstrip('_'), parent_dir 
        return "", parent_dir


class BatchAccessor:
    """Batch accessor supporting dataset[:]['key'] syntax with DataLoader-like stacking."""
    
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
    
    def __getitem__(self, key):
        """Support dataset[:]['key'] syntax and stack tensors automatically."""
        return self._get_key(key)
    
    def _get_key(self, key):
        """Return one key and stack tensor values automatically."""
        # Normalize supported index types.
        if isinstance(self.indices, slice):
            indices = range(*self.indices.indices(len(self.dataset)))
        else:
            indices = self.indices
        
        result = []
        for i in indices:
            item = self.dataset._get_single_item(i)
            if key in item:
                result.append(item[key])
            else:
                raise KeyError(f"Key '{key}' not found in dataset item")
        
        # Stack tensors to mimic DataLoader behavior.
        if result and isinstance(result[0], torch.Tensor):
            return torch.stack(result)
        else:
            # Return non-tensor values such as category_name/object_name as a list.
            return result
    
    def __iter__(self):
        """Iterate over full item dictionaries."""
        if isinstance(self.indices, slice):
            indices = range(*self.indices.indices(len(self.dataset)))
        else:
            indices = self.indices
            
        for i in indices:
            yield self.dataset._get_single_item(i)
    
    def __len__(self):
        """Return batch length."""
        if isinstance(self.indices, slice):
            return len(range(*self.indices.indices(len(self.dataset))))
        else:
            return len(self.indices)
    
    def __getattr__(self, name):
        """Support dataset[:].image_seq syntax."""
        valid_keys = ['image_seq', 'action_seq', 'transform_seq', 'seq_index', 
                     'category_name', 'object_name', 'category']
        
        if name in valid_keys:
            return self._get_key(name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def keys(self):
        """Return available keys, mimicking a batch dictionary."""
        return ['image_seq', 'action_seq', 'transform_seq', 'seq_index', 
                'category_name', 'object_name', 'category']
    
    def items(self):
        """Return all key-value pairs, mimicking a batch dictionary."""
        return [(key, self._get_key(key)) for key in self.keys()]
    
    def values(self):
        """Return all values, mimicking a batch dictionary."""
        return [self._get_key(key) for key in self.keys()]
    

def visualize_sequence_with_actions(images, actions, seq_idx=0, action_types=['rotation', 'scale', 'translation_x', 'translation_y', 'plane_rotation']):
    """
    Visualize one sequence, with each title showing the corresponding action.
    
    Args:
        images: torch.Tensor, shape (batch_size, seq_len, 3, H, W)
        actions: torch.Tensor, shape (batch_size, seq_len-1, action_dim)
        seq_idx: sequence index to visualize.
        action_types: action type names.
    """
    
    # Select the requested sequence.
    # seq_images = images[seq_idx]  # (seq_len, 3, H, W)
    # seq_actions = actions[seq_idx]  # (seq_len-1, action_dim)

    seq_images = images  # (seq_len, 3, H, W)
    seq_actions = actions  # (seq_len-1, action_dim)
    
    seq_len = seq_images.shape[0]
    
    # Create subplots.
    cols = min(seq_len, 10)
    rows = (seq_len + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.5))
    
    # Keep axes two-dimensional for uniform indexing.
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(seq_len):
        row = i // cols
        col = i % cols
        
        # Convert image format for display.
        img = seq_images[i].permute(1, 2, 0)  # (H, W, 3)
        
        # Convert from [-1, 1] to [0, 1] when needed.
        if img.min() >= -1 and img.max() <= 1:
            img = (img + 1) / 2
        
        # Clamp to display range.
        img = torch.clamp(img, 0, 1)
        
        # Show image.
        axes[row, col].imshow(img.cpu().numpy())
        axes[row, col].axis('off')
        
        # Set title.
        if i == 0:
            # The first image shows the initial state.
            title = f"Frame {i}\nInitial State"
        else:
            # Later images show the action from the previous frame.
            action = seq_actions[i-1]
            
            # Format the action string.
            action_str = []
            for j, action_type in enumerate(action_types):
                if j < len(action):
                    if action_type == 'rotation':
                        action_str.append(f"rt: {action[j]:.1f}°")
                    elif action_type == 'scale':
                        action_str.append(f"sc: {action[j]:.2f}")
                    elif action_type == 'translation_x':
                        action_str.append(f"tx: {action[j]:.1f}")
                    elif action_type == 'translation_y':
                        action_str.append(f"ty: {action[j]:.1f}")
                    elif action_type == 'plane_rotation':
                        action_str.append(f"pr: {action[j]:.1f}°")
            
            title = f"Frame {i}\n" + ", ".join(action_str)
        
        axes[row, col].set_title(title, fontsize=8, pad=5)
    
    # Hide unused subplots.
    for i in range(seq_len, rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show() 


class ProcGenDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 seq_len=10, 
                 stride=1, 
                 transform=None, 
                 split_ratio=0.8,  # Used only when the dataset has no explicit split directories.
                 mode='train',    # 'train', 'test', 'val', 'all'
                 seed=42,
                 return_goal=True,
                 cache_metadata=False):
        
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.split_ratio = split_ratio
        self.mode = mode.lower()
        self.seed = seed
        self.return_goal = return_goal
        self.cache_metadata = cache_metadata
        
        # Treat validation as test for split selection.
        if self.mode == 'val':
            self.mode = 'test'
        
        self.samples = [] 
        self.json_cache = {} 
        # self.split_info = {}
        
        self._build_index()

    def _get_sequence_folders(self):
        # 1. Scan candidate sequence folders.
        all_folders = []
        for root, _, files in os.walk(self.root_dir):
            if 'meta.json' in files:
                all_folders.append(root)
        
        # 2. Classify explicit train/test folders.
        explicit_train = []
        explicit_test = []
        ambiguous = [] 
        
        for folder in all_folders:
            rel_path = os.path.relpath(folder, self.root_dir)
            path_parts = rel_path.split(os.sep)
            path_parts_lower = [p.lower() for p in path_parts]
            
            # Strict directory-name matching.
            is_train_dir = 'train' in path_parts_lower or 'training' in path_parts_lower
            is_test_dir = 'test' in path_parts_lower or 'testing' in path_parts_lower or 'val' in path_parts_lower or 'validation' in path_parts_lower
            
            if is_train_dir:
                explicit_train.append(folder)
            elif is_test_dir:
                explicit_test.append(folder)
            else:
                ambiguous.append(folder)
        
        # 3. Split ambiguous folders deterministically.
        ambiguous.sort()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(ambiguous)
        
        split_idx = int(len(ambiguous) * self.split_ratio)
        ambiguous_train = ambiguous[:split_idx]
        ambiguous_test = ambiguous[split_idx:]
        
        final_folders = []
        source_stats = {}
        
        if self.mode == 'all':
            final_folders = explicit_train + explicit_test + ambiguous
            source_stats = {'explicit': len(explicit_train)+len(explicit_test), 'random': len(ambiguous)}
            
        elif self.mode == 'train':
            final_folders = explicit_train + ambiguous_train
            source_stats = {'explicit': len(explicit_train), 'random': len(ambiguous_train)}
            
        elif self.mode == 'test':
            final_folders = explicit_test + ambiguous_test
            source_stats = {'explicit': len(explicit_test), 'random': len(ambiguous_test)}
        
        # Optional split diagnostics.
        # self.split_info = {
        #     'total_folders': len(final_folders),
        #     'source_breakdown': source_stats,
        #     'ambiguous_pool_size': len(ambiguous),
        #     'explicit_pool_size': len(explicit_train) + len(explicit_test)
        # }
        
        return final_folders

    def _build_index(self):
        target_folders = self._get_sequence_folders()
        
        for folder_path in target_folders:
            json_path = os.path.join(folder_path, 'meta.json')
            
            try:
                with open(json_path, 'r') as f:
                    meta_data = json.load(f)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue

            if self.cache_metadata:
                self.json_cache[folder_path] = meta_data
            
            total_frames = len(meta_data)
            rel_path = os.path.relpath(folder_path, self.root_dir)

            max_start_idx = total_frames - self.seq_len
            if max_start_idx < 0: continue
            
            for start_idx in range(0, max_start_idx + 1, self.stride):
                self.samples.append({
                    'folder_path': folder_path,
                    'start_idx': start_idx,
                    'rel_path': rel_path,
                    'meta_ref': meta_data if self.cache_metadata else None
                })


    def _process_action(self, action_raw):
        # 1. List/array -> tensor.
        if isinstance(action_raw, (list, tuple, np.ndarray)):
            # Procgen can provide [int], while Robodesk can provide [float, ...].
            # Normalize actions to float vectors for model input.
            return torch.tensor(action_raw, dtype=torch.float32)

        # 2. Dict action format (e.g., Minecraft) -> tensor.
        elif isinstance(action_raw, dict):
            fwd = 1.0 if action_raw.get('forward', False) else 0.0
            jump = 1.0 if action_raw.get('jump', False) else 0.0
            cam = action_raw.get('camera', [0.0, 0.0])
            if not isinstance(cam, list) or len(cam) < 2: cam = [0.0, 0.0]
            return torch.tensor([fwd, jump, float(cam[0]), float(cam[1])], dtype=torch.float32)

        # 3. Scalar action -> tensor, compatible with Procgen-style integer actions.
        elif isinstance(action_raw, (int, float, np.number)):
             return torch.tensor([float(action_raw)], dtype=torch.float32)

        else:
            return torch.zeros(1, dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        folder_path = item['folder_path']
        start_idx = item['start_idx']
        
        if self.cache_metadata:
            full_meta = item['meta_ref']
        else:
            with open(os.path.join(folder_path, 'meta.json'), 'r') as f:
                full_meta = json.load(f)
        
        seq_meta = full_meta[start_idx : start_idx + self.seq_len]
        
        # Collect metadata.
        actions, goals = [], []
        # Keep common state fields.
        xs, ys, zs, yaws, pitches = [], [], [], [], []
        
        for frame_data in seq_meta:
            xs.append(frame_data.get('x', 0.0))
            ys.append(frame_data.get('y', 0.0))
            zs.append(frame_data.get('z', 0.0))
            yaws.append(frame_data.get('yaw', 0.0))
            pitches.append(frame_data.get('pitch', 0.0))
            
            # Action
            actions.append(self._process_action(frame_data.get('action', [])))
            
            # Goal
            if self.return_goal:
                g_raw = frame_data.get('goal', [0.0, 0.0, 0.0])
                if isinstance(g_raw, dict):
                     g_vec = [g_raw.get('x', 0.0), g_raw.get('y', 0.0), g_raw.get('z', 0.0)]
                elif isinstance(g_raw, (list, tuple, np.ndarray)):
                    g_vec = g_raw
                else:
                    g_vec = [0.0, 0.0, 0.0]
                goals.append(torch.tensor(g_vec, dtype=torch.float32))

        # Load images.
        images_list = []
        for i in range(self.seq_len):
            curr_idx = start_idx + i
            img_path = os.path.join(folder_path, f"{curr_idx:05d}.jpg")
            try:
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    images_list.append(img)
            except Exception:
                images_list.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        # Transform
        if self.transform:
            processed = [self.transform(img) for img in images_list]
            images_tensor = torch.stack(processed)
        else:
            tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float()/255.0 for img in images_list]
            images_tensor = torch.stack(tensors)

        result = {
            'images': images_tensor,
            'action': torch.stack(actions),
            'path': item['rel_path'],
            # Optional auxiliary state.
            'state': torch.tensor([xs, ys, zs, yaws, pitches], dtype=torch.float32).T 
        }
        if self.return_goal and len(goals) > 0:
            result['goal'] = torch.stack(goals)

        return result



class LoopNavDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 seq_len=10, 
                 stride=1, 
                 transform=None, 
                 split_ratio=0.8,
                 mode='train',
                 seed=42,
                 return_goal=True,
                 cache_metadata=False,  # Default to False to save memory
                 max_cached_meta=1000):  # Limit cache size
        """
        LoopNav Dataset with memory-efficient lazy loading.
        
        Args:
            root_dir: Root directory containing the dataset
            seq_len: Sequence length
            stride: Stride for sequence sampling
            transform: Image transform
            split_ratio: Train/test split ratio
            mode: 'train', 'val', 'test', or 'all'
            seed: Random seed for reproducibility
            return_goal: Whether to return goal information
            cache_metadata: Whether to cache metadata (set False for large datasets)
            max_cached_meta: Maximum number of metadata entries to cache (LRU-style)
        
        Returns dict with:
            images: shape: [seq_len, C, H, W]
            x, y, z, yaw, pitch: shape: [seq_len] absolute positions and orientations
            actions: shape: [seq_len, 4] (forward, jump, camera_h, camera_v)
            goal: shape: [seq_len, 3] (optional) absolute goal positions
            path: relative path from dataset root_dir to the sequence folder
        """
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.split_ratio = split_ratio
        self.mode = mode
        self.seed = seed
        self.return_goal = return_goal
        self.cache_metadata = cache_metadata
        self.max_cached_meta = max_cached_meta
        
        self.samples = [] 
        self.json_cache = {}  # LRU-style cache
        self._cache_order = []  # Track cache order for LRU eviction

        self._build_index()

    def _get_sequence_folders(self):
        # Find folders containing meta.json; each folder is one converted video.
        all_folders = []
        for root, _, files in os.walk(self.root_dir):
            if 'meta.json' in files:
                all_folders.append(root)
        
        # Sort and shuffle deterministically.
        all_folders.sort()
        rng = np.random.default_rng(self.seed)
        rng.shuffle(all_folders)
        
        # Split into train/test partitions.
        if self.mode == 'all': return all_folders
        
        split_idx = int(len(all_folders) * self.split_ratio)
        if self.mode == 'train':
            return all_folders[:split_idx]
        elif self.mode in ['test', 'val']:
            return all_folders[split_idx:]
        else:
            raise ValueError("Unknown mode")

    def _build_index(self):
        target_folders = self._get_sequence_folders()
        
        for folder_path in target_folders:
            json_path = os.path.join(folder_path, 'meta.json')
            
            try:
                with open(json_path, 'r') as f:
                    meta_data = json.load(f)
            except Exception as e:
                print(f"Error loading {json_path}: {e}")
                continue

            total_frames = len(meta_data)
            
            # Store metadata in cache if enabled (with LRU eviction)
            if self.cache_metadata:
                self._cache_metadata(folder_path, meta_data)
            
            # rel_path for reference
            rel_path = os.path.relpath(folder_path, self.root_dir)

            max_start_idx = total_frames - self.seq_len
            if max_start_idx < 0: 
                continue
            
            for start_idx in range(0, max_start_idx + 1, self.stride):
                # Don't store meta_ref directly - load on demand to save memory
                self.samples.append({
                    'folder_path': folder_path,
                    'start_idx': start_idx,
                    'rel_path': rel_path,
                    'total_frames': total_frames,  # Store length instead of full data
                })
        
    
    def _cache_metadata(self, folder_path: str, meta_data: list):
        """Cache metadata with LRU eviction to limit memory usage"""
        if folder_path in self.json_cache:
            # Move to end (most recently used)
            self._cache_order.remove(folder_path)
            self._cache_order.append(folder_path)
            return
        
        # Evict oldest entries if cache is full
        while len(self.json_cache) >= self.max_cached_meta and self._cache_order:
            oldest = self._cache_order.pop(0)
            del self.json_cache[oldest]
        
        self.json_cache[folder_path] = meta_data
        self._cache_order.append(folder_path)
    
    def _get_metadata(self, folder_path: str) -> list:
        """Get metadata with lazy loading and LRU caching"""
        if folder_path in self.json_cache:
            # Move to end (most recently used)
            if folder_path in self._cache_order:
                self._cache_order.remove(folder_path)
                self._cache_order.append(folder_path)
            return self.json_cache[folder_path]
        
        # Load from disk
        json_path = os.path.join(folder_path, 'meta.json')
        with open(json_path, 'r') as f:
            meta_data = json.load(f)
        
        # Cache if enabled
        if self.cache_metadata:
            self._cache_metadata(folder_path, meta_data)
        
        return meta_data

    def _process_action(self, action_dict):
        fwd = 1.0 if action_dict.get('forward', False) else 0.0
        jump = 1.0 if action_dict.get('jump', False) else 0.0
        cam = action_dict.get('camera', [0.0, 0.0])
        if not isinstance(cam, list) or len(cam) < 2: cam = [0.0, 0.0]
        return torch.tensor([fwd, jump, float(cam[0]), float(cam[1])], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        folder_path = item['folder_path']
        start_idx = item['start_idx']
        
        # --- 1. Meta Data (lazy loading with LRU cache) ---
        full_meta = self._get_metadata(folder_path)
        seq_meta = full_meta[start_idx : start_idx + self.seq_len]
        
        # Parse metadata fields.
        xs, ys, zs, yaws, pitches, actions, goals = [], [], [], [], [], [], []
        for frame_data in seq_meta:
            xs.append(frame_data.get('x', 0.0))
            ys.append(frame_data.get('y', 0.0))
            zs.append(frame_data.get('z', 0.0))
            yaws.append(frame_data.get('yaw', 0.0))
            pitches.append(frame_data.get('pitch', 0.0))
            actions.append(self._process_action(frame_data.get('action', {})))
            
            if self.return_goal:
                g_raw = frame_data.get('goal', [0.0, 0.0, 0.0])
                if isinstance(g_raw, dict):
                     g_vec = [g_raw.get('x', 0.0), g_raw.get('y', 0.0), g_raw.get('z', 0.0)]
                elif isinstance(g_raw, (list, tuple)):
                    g_vec = g_raw
                else:
                    g_vec = [0.0, 0.0, 0.0]
                goals.append(torch.tensor(g_vec, dtype=torch.float32))

        # --- 2. Load images ---
        images_list = []
        for i in range(self.seq_len):
            curr_idx = start_idx + i
            # Format filename as 00000.jpg.
            img_name = f"{curr_idx:05d}.jpg" 
            img_path = os.path.join(folder_path, img_name)
            
            try:
                # PIL integrates cleanly with torchvision transforms.
                with Image.open(img_path) as img:
                    img = img.convert('RGB')
                    images_list.append(img)
            except Exception as e:
                # On read failure, use a black placeholder image.
                # print(f"Error reading {img_path}: {e}")
                images_list.append(Image.new('RGB', (224, 224), (0, 0, 0)))

        # --- 3. Transform ---
        if self.transform:
            # Transform each PIL image independently.
            processed = [self.transform(img) for img in images_list]
            images_tensor = torch.stack(processed)
        else:
            # Basic ToTensor conversion.
            tensors = [torch.from_numpy(np.array(img)).permute(2, 0, 1).float()/255.0 for img in images_list]
            images_tensor = torch.stack(tensors)

        result = {
            'images': images_tensor,
            'x': torch.tensor(xs, dtype=torch.float32),
            'y': torch.tensor(ys, dtype=torch.float32),
            'z': torch.tensor(zs, dtype=torch.float32),
            'yaw': torch.tensor(yaws, dtype=torch.float32),
            'pitch': torch.tensor(pitches, dtype=torch.float32),
            'action': torch.stack(actions),
            'path': item['rel_path']
        }
        if self.return_goal and len(goals) > 0:
            result['goal'] = torch.stack(goals)

        return result
    
    def clear_cache(self):
        """Clear the metadata cache to free memory"""
        self.json_cache.clear()
        self._cache_order.clear()
        import gc
        gc.collect()
    
    def __del__(self):
        """Cleanup when dataset is deleted"""
        try:
            self.clear_cache()
        except Exception:
            pass  # Ignore cleanup errors during deletion


# copy from nwm/misc.py
def angle_difference(theta1, theta2):
    delta_theta = theta2 - theta1    
    delta_theta = delta_theta - 2 * np.pi * np.floor((delta_theta + np.pi) / (2 * np.pi))    
    return delta_theta

def get_data_path(data_folder: str, f: str, time: int, data_type: str = "image"):
    data_ext = {
        "image": ".jpg",
        # add more data types here
    }
    return os.path.join(data_folder, f, f"{str(time)}{data_ext[data_type]}")

def get_delta_np(actions):
    # append zeros to first action (unbatched)
    ex_actions = np.concatenate((np.zeros((1, actions.shape[1])), actions), axis=0)
    delta = ex_actions[1:] - ex_actions[:-1]
    
    return delta

def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata

def to_local_coords(
    positions: np.ndarray, curr_pos: np.ndarray, curr_yaw: float
) -> np.ndarray:
    """
    Convert positions to local coordinates

    Args:
        positions (np.ndarray): positions to convert
        curr_pos (np.ndarray): current position
        curr_yaw (float): current yaw
    Returns:
        np.ndarray: positions in local coordinates
    """
    rotmat = yaw_rotmat(curr_yaw)
    if positions.shape[-1] == 2:
        rotmat = rotmat[:2, :2]
    elif positions.shape[-1] == 3:
        pass
    else:
        raise ValueError

    return (positions - curr_pos).dot(rotmat)

def yaw_rotmat(yaw: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
    )

class BaseDataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        self.goals_per_obs = goals_per_obs


        traj_names_file = os.path.join(data_split_folder, traj_names)
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.distance_categories = list(range(min_dist_cat, max_dist_cat + 1))
        self.min_dist_cat = self.distance_categories[0]
        self.max_dist_cat = self.distance_categories[-1]
        self.len_traj_pred = len_traj_pred
        self.traj_stride = traj_stride

        self.context_size = context_size
        self.normalize = normalize

        # load data/data_config.yaml
        # with open("config/data_config.yaml", "r") as f:
        #     all_data_config = yaml.safe_load(f)

        all_data_config= {
                        'action_stats': {
                            'min': [-2.5, -4],  # [min_dx, min_dy]
                            'max': [5, 4],      # [max_dx, max_dy]
                        },
                        'recon': {
                            'metric_waypoint_spacing': 0.25,
                        },
                        'scand': {
                            'metric_waypoint_spacing': 0.38,
                        },
                        'tartan_drive': {
                            'metric_waypoint_spacing': 0.72,
                        },
                        'go_stanford': {
                            'metric_waypoint_spacing': 0.12,
                        },
                        'sacson': {
                            'metric_waypoint_spacing': 0.255,
                        },
                    }

        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.data_config = all_data_config[self.dataset_name]
        self.transform = transform
        self._load_index(predefined_index)
        self.ACTION_STATS = {}
        for key in all_data_config['action_stats']:
            self.ACTION_STATS[key] = np.expand_dims(all_data_config['action_stats'][key], axis=0)

    def _load_index(self, predefined_index) -> None:
        """
        Generates a list of tuples of (obs_traj_name, goal_traj_name, obs_time, goal_time) for each observation in the dataset
        """
        if predefined_index:
            with open(predefined_index, "rb") as f:
                self.index_to_data = pickle.load(f)
                return
        else:
            self.index_to_data, self.goals_index = self._build_index()

    def _build_index(self, use_tqdm: bool = False):
        """
        Build an index consisting of tuples (trajectory name, time, max goal distance)
        """
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])
            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size - 1
            end_time = traj_len - self.len_traj_pred
            for curr_time in range(begin_time, end_time, self.traj_stride):
                max_goal_distance = min(self.max_dist_cat, traj_len - curr_time - 1)
                min_goal_distance = max(self.min_dist_cat, -curr_time)
                samples_index.append((traj_name, curr_time, min_goal_distance, max_goal_distance))

        return samples_index, goals_index
  
    def _get_trajectory(self, trajectory_name):
        with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
            traj_data = pickle.load(f)
        for k,v in traj_data.items():
            traj_data[k] = v.astype('float')
        return traj_data

    def __len__(self) -> int:
        return len(self.index_to_data)

    def _compute_actions(self, traj_data, curr_time, goal_time):
        start_index = curr_time
        end_index = curr_time + self.len_traj_pred + 1
        yaw = traj_data["yaw"][start_index:end_index]
        positions = traj_data["position"][start_index:end_index]
        goal_pos = traj_data["position"][goal_time]
        goal_yaw = traj_data["yaw"][goal_time]

        if len(yaw.shape) == 2:
            yaw = yaw.squeeze(1)

        if yaw.shape != (self.len_traj_pred + 1,):
            raise ValueError("is used?")
            # const_len = self.len_traj_pred + 1 - yaw.shape[0]
            # yaw = np.concatenate([yaw, np.repeat(yaw[-1], const_len)])
            # positions = np.concatenate([positions, np.repeat(positions[-1][None], const_len, axis=0)], axis=0)

        waypoints_pos = to_local_coords(positions, positions[0], yaw[0])
        waypoints_yaw = angle_difference(yaw[0], yaw)
        actions = np.concatenate([waypoints_pos, waypoints_yaw.reshape(-1, 1)], axis=-1)
        actions = actions[1:]
        
        goal_pos = to_local_coords(goal_pos, positions[0], yaw[0])
        goal_yaw = angle_difference(yaw[0], goal_yaw)
        
        if self.normalize:
            actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
            goal_pos[:, :2] /= self.data_config["metric_waypoint_spacing"]
        
        goal_pos = np.concatenate([goal_pos, goal_yaw.reshape(-1, 1)], axis=-1)
        return actions, goal_pos    

class TrainingDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str = 'traj_names.txt',
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)


    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1, size=(self.goals_per_obs))
            goal_time = (curr_time + goal_offset).astype('int')
            rel_time = (goal_offset).astype('float')/(128.) # TODO: refactor, currently a fixed const

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            context = [(f_curr, t) for t in context_times] + [(f_curr, t) for t in goal_time]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])

            # Load other trajectory data
            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            _, goal_pos = self._compute_actions(curr_traj_data, curr_time, goal_time)
            goal_pos[:, :2] = normalize_data(goal_pos[:, :2], self.ACTION_STATS)

            return (
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
                torch.as_tensor(rel_time, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)

class EvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str = 'traj_names.txt',
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)
  
    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, _, _ = self.index_to_data[i]
            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))
            pred_times = list(range(curr_time + 1, curr_time + self.len_traj_pred + 1))
            
            context = [(f_curr, t) for t in context_times]
            pred = [(f_curr, t) for t in pred_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            pred_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in pred])

            curr_traj_data = self._get_trajectory(f_curr)

            # Compute actions
            actions, _ = self._compute_actions(curr_traj_data, curr_time, np.array([curr_time+1])) # last argument is dummy goal
            actions[:, :2] = normalize_data(actions[:, :2], self.ACTION_STATS)
            delta = get_delta_np(actions)

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(pred_image, dtype=torch.float32),
                torch.as_tensor(delta, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)
        
class TrajectoryEvalDataset(BaseDataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        min_dist_cat: int,
        max_dist_cat: int,
        len_traj_pred: int,
        traj_stride: int, 
        context_size: int,
        transform: object,
        traj_names: str,
        normalize: bool = True,
        predefined_index: list = None,
        goals_per_obs: int = 1,
    ):
        super().__init__(data_folder, data_split_folder, dataset_name, image_size, min_dist_cat, max_dist_cat,
            len_traj_pred, traj_stride, context_size, transform, traj_names, normalize, predefined_index, goals_per_obs)

   
    def _sample_goal(self, trajectory_name, curr_time, min_goal_dist, max_goal_dist):
        """
        Sample a goal from the future in the same trajectory.
        Returns: (trajectory_name, goal_time, goal_is_negative)
        """
        goal_offset = np.random.randint(min_goal_dist, max_goal_dist + 1)
        goal_time = curr_time + int(goal_offset)
        return trajectory_name, goal_time, False

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        try:
            f_curr, curr_time, min_goal_dist, max_goal_dist = self.index_to_data[i]
            f_goal, goal_time, _ = self._sample_goal(f_curr, curr_time, min_goal_dist, max_goal_dist)

            context_times = list(range(curr_time - self.context_size + 1, curr_time + 1))           
            context = [(f_curr, t) for t in context_times]

            obs_image = torch.stack([self.transform(Image.open(get_data_path(self.data_folder, f, t))) for f, t in context])
            goal_image = self.transform(Image.open(get_data_path(self.data_folder, f_goal, goal_time))).unsqueeze(0)
            curr_traj_data = self._get_trajectory(f_curr)

            actions, goal_pos = self._compute_actions(curr_traj_data, curr_time, np.array([goal_time]))

            return (
                torch.tensor([i], dtype=torch.float32), # for logging purposes
                torch.as_tensor(obs_image, dtype=torch.float32),
                torch.as_tensor(goal_image, dtype=torch.float32),
                torch.as_tensor(actions, dtype=torch.float32),
                torch.as_tensor(goal_pos, dtype=torch.float32),
            )
        except Exception as e:
            print(f"Exception in {self.dataset_name}", e)
            raise Exception(e)


class OpenXProcessedDataset(Dataset):
    """
    PyTorch Dataset for Open-X-Embodiment data pre-processed into folder + frames format.
    
    Memory-efficient implementation with lazy loading of frame file lists.
    
    Args:
        data_dir: Root directory containing processed Open-X-Embodiment datasets
        split: 'train' or 'val' (will split by ratio if no explicit split exists)
        seq_len: Number of frames per sequence
        transform: Optional transform to apply to images
        frame_skip: Skip frames (temporal downsampling), default 1 means no skip
        dataset_names: List of specific datasets to load (e.g., ['fractal20220817_data'])
        max_episodes_per_dataset: Maximum number of episodes to load per dataset
        train_ratio: Ratio of data to use for training (default 0.9)
        seed: Random seed for train/val split
        return_instruction: Whether to return instruction text along with images
        lazy_frame_list: If True, don't store frame file lists in memory (saves memory for large datasets)
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        seq_len: int = 16,
        transform: Optional[Callable] = None,
        frame_skip: int = 1,
        dataset_names: Optional[List[str]] = None,
        max_episodes_per_dataset: Optional[int] = None,
        train_ratio: float = 0.9,
        seed: int = 42,
        return_instruction: bool = False,
        lazy_frame_list: bool = True  # Default to lazy loading for memory efficiency
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.seq_len = seq_len
        self.transform = transform
        self.frame_skip = frame_skip
        self.dataset_names = dataset_names
        self.max_episodes_per_dataset = max_episodes_per_dataset
        self.train_ratio = train_ratio
        self.seed = seed
        self.return_instruction = return_instruction
        self.lazy_frame_list = lazy_frame_list
        
        # Storage for episode information (minimal footprint)
        self.episodes: List[Dict[str, Any]] = []
        self.sequence_index: List[Tuple[int, int, int]] = []  # (episode_idx, start_frame, end_frame)
        
        # Load episodes
        self._discover_and_load_episodes()
        
        # Build sequence index
        self._build_sequence_index()
        
    
    def _discover_and_load_episodes(self):
        """Discover and load all episodes from the data directory"""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {self.data_dir}")
        
        # Find all dataset folders (pattern: *_videos)
        dataset_folders = []
        
        # Check if data_dir itself is a dataset folder
        if self._is_dataset_folder(self.data_dir):
            dataset_folders.append(self.data_dir)
        else:
            # Look for *_videos folders
            for item in self.data_dir.iterdir():
                if item.is_dir():
                    if self._is_dataset_folder(item):
                        dataset_folders.append(item)
                    # Also check one level deeper
                    for sub_item in item.iterdir():
                        if sub_item.is_dir() and self._is_dataset_folder(sub_item):
                            dataset_folders.append(sub_item)
        
        # Filter by dataset_names if specified
        if self.dataset_names:
            filtered_folders = []
            for folder in dataset_folders:
                folder_name = folder.name
                # Check if any of the dataset_names matches
                for name in self.dataset_names:
                    if name in folder_name or folder_name.startswith(name):
                        filtered_folders.append(folder)
                        break
            dataset_folders = filtered_folders
        
        # Load episodes from each dataset folder
        all_episodes = []
        for dataset_folder in dataset_folders:
            episodes = self._load_dataset_folder(dataset_folder)
            all_episodes.extend(episodes)
        
        # Apply train/val split
        if all_episodes:
            rng = np.random.RandomState(self.seed)
            indices = np.arange(len(all_episodes))
            rng.shuffle(indices)
            
            split_idx = int(len(indices) * self.train_ratio)
            
            if self.split == 'train':
                selected_indices = indices[:split_idx]
            else:  # val
                selected_indices = indices[split_idx:]
            
            self.episodes = [all_episodes[i] for i in selected_indices]
        else:
            print(f"Warning: No episodes found in {self.data_dir}")
    
    def _is_dataset_folder(self, folder: Path) -> bool:
        """Check if a folder is a valid dataset folder with video_X subfolders"""
        # Look for video_* folders
        video_folders = list(folder.glob("video_*"))
        return len(video_folders) > 0
    
    def _load_dataset_folder(self, dataset_folder: Path) -> List[Dict[str, Any]]:
        """Load all episodes from a dataset folder with memory-efficient lazy loading"""
        episodes = []
        
        # Load instructions if available
        instructions = self._load_instructions(dataset_folder)
        
        # Find all video folders
        video_folders = sorted(
            dataset_folder.glob("video_*"),
            key=lambda x: int(x.name.split('_')[1]) if x.name.split('_')[1].isdigit() else 0
        )
        
        for video_folder in video_folders:
            if not video_folder.is_dir():
                continue
            
            # Get video index
            video_idx = self._extract_video_index(video_folder.name)
            
            # Count frames (but don't store full list if lazy_frame_list is True)
            frame_files = self._get_frame_files(video_folder)
            num_frames = len(frame_files)
            
            if num_frames == 0:
                continue
            
            # Get instruction for this video
            instruction = instructions.get(video_idx, "")
            
            episode_info = {
                'video_folder': video_folder,
                'dataset_name': dataset_folder.name,
                'video_idx': video_idx,
                'num_frames': num_frames,
                'instruction': instruction
            }
            
            # Only store frame_files if not using lazy loading
            if not self.lazy_frame_list:
                episode_info['frame_files'] = frame_files
            
            episodes.append(episode_info)
            
            if self.max_episodes_per_dataset and len(episodes) >= self.max_episodes_per_dataset:
                break
        
        return episodes
    
    def _load_instructions(self, dataset_folder: Path) -> Dict[int, str]:
        """Load instructions from the instruction file"""
        instructions = {}
        
        # Look for instruction files
        instruction_files = list(dataset_folder.glob("*.txt"))
        
        for instruction_file in instruction_files:
            if instruction_file.name == 'state.txt':
                continue  # Skip state file
            
            try:
                with open(instruction_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        
                        # Parse "Video X Instruction: ..." format
                        match = re.match(r'Video\s+(\d+)\s+Instruction:\s*(.+)', line)
                        if match:
                            video_idx = int(match.group(1))
                            instruction_text = match.group(2)
                            instructions[video_idx] = instruction_text
            except Exception as e:
                print(f"Warning: Failed to load instructions from {instruction_file}: {e}")
        
        return instructions
    
    def _extract_video_index(self, folder_name: str) -> int:
        """Extract video index from folder name like 'video_0'"""
        parts = folder_name.split('_')
        for part in parts:
            if part.isdigit():
                return int(part)
        return 0
    
    def _get_frame_files(self, video_folder: Path) -> List[Path]:
        """Get sorted list of frame files in a video folder"""
        # Support both frame_X.png and frame_X.jpg formats
        frame_files = []
        
        for ext in ['png', 'jpg', 'jpeg']:
            frame_files.extend(video_folder.glob(f"frame_*.{ext}"))
        
        # Sort by frame index
        frame_files = sorted(
            frame_files,
            key=lambda x: int(re.search(r'frame_(\d+)', x.stem).group(1)) if re.search(r'frame_(\d+)', x.stem) else 0
        )
        
        return frame_files
    
    def _build_sequence_index(self):
        """Build an index of all valid sequences"""
        self.sequence_index = []
        
        for episode_idx, episode_info in enumerate(self.episodes):
            num_frames = episode_info['num_frames']
            
            # Calculate effective sequence length considering frame_skip
            effective_seq_len = self.seq_len * self.frame_skip
            
            if num_frames < effective_seq_len:
                # If episode is too short, still include it as a single (potentially padded) sequence
                self.sequence_index.append((episode_idx, 0, num_frames))
                continue
            
            # Create sequences with sliding window
            # Stride equals effective sequence length for non-overlapping sequences
            stride = effective_seq_len
            
            for start_frame in range(0, num_frames - effective_seq_len + 1, stride):
                end_frame = start_frame + effective_seq_len
                self.sequence_index.append((episode_idx, start_frame, end_frame))
    
    def _load_frames(self, episode_idx: int, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load frames for a specific episode and frame range with lazy loading support"""
        episode_info = self.episodes[episode_idx]
        num_frames = episode_info['num_frames']
        
        # Get frame files (lazy load if needed)
        if self.lazy_frame_list or 'frame_files' not in episode_info:
            frame_files = self._get_frame_files(episode_info['video_folder'])
        else:
            frame_files = episode_info['frame_files']
        
        # Generate frame indices with frame_skip
        frame_indices = list(range(start_frame, min(end_frame, num_frames), self.frame_skip))
        
        # Pad indices if needed
        while len(frame_indices) < self.seq_len:
            # Repeat last frame index
            frame_indices.append(frame_indices[-1] if frame_indices else 0)
        
        # Truncate if needed
        frame_indices = frame_indices[:self.seq_len]
        
        # Load images
        images = []
        placeholder_shape = None
        
        for frame_idx in frame_indices:
            if frame_idx < len(frame_files):
                img_path = frame_files[frame_idx]
                try:
                    img = Image.open(img_path).convert('RGB')
                    
                    if self.transform is not None:
                        img = self.transform(img)
                    else:
                        # Default: convert to tensor
                        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                    
                    if placeholder_shape is None:
                        placeholder_shape = img.shape
                    
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Failed to load image {img_path}: {e}")
                    # Create a placeholder
                    if images:
                        images.append(images[-1].clone())
                    elif placeholder_shape:
                        images.append(torch.zeros(placeholder_shape))
                    else:
                        images.append(torch.zeros(3, 224, 224))
            else:
                # Repeat last image
                if images:
                    images.append(images[-1].clone())
                elif placeholder_shape:
                    images.append(torch.zeros(placeholder_shape))
                else:
                    images.append(torch.zeros(3, 224, 224))
        
        # Stack to create (T, C, H, W) tensor
        images = torch.stack(images, dim=0)
        
        return images
    
    def __len__(self) -> int:
        return len(self.sequence_index)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Dict[str, Any]]:
        """Get a sequence of frames"""
        episode_idx, start_frame, end_frame = self.sequence_index[idx]
        
        # Load frames
        images = self._load_frames(episode_idx, start_frame, end_frame)
        
        if self.return_instruction:
            instruction = self.episodes[episode_idx]['instruction']
            return {
                'observations': images,
                'instruction': instruction,
                'dataset_name': self.episodes[episode_idx]['dataset_name']
            }
        
        return images
    
    def get_episode_info(self, idx: int) -> Dict[str, Any]:
        """Get metadata for a specific sequence index"""
        episode_idx, start_frame, end_frame = self.sequence_index[idx]
        episode_info = self.episodes[episode_idx]
        
        return {
            'dataset_name': episode_info['dataset_name'],
            'video_idx': episode_info['video_idx'],
            'num_frames': episode_info['num_frames'],
            'start_frame': start_frame,
            'end_frame': end_frame,
            'instruction': episode_info['instruction']
        }
    
    def clear_cache(self):
        """Clear any cached data to free memory"""
        # For lazy_frame_list=False, frame_files are stored in episodes
        # This method can be called to free that memory if needed
        if not self.lazy_frame_list:
            for episode in self.episodes:
                if 'frame_files' in episode:
                    del episode['frame_files']
            self.lazy_frame_list = True  # Switch to lazy mode
            import gc
            gc.collect()


def create_openx_processed_dataloader(
    data_dir: str,
    split: str = 'train',
    seq_len: int = 16,
    batch_size: int = 32,
    transform: Optional[Callable] = None,
    frame_skip: int = 1,
    dataset_names: Optional[List[str]] = None,
    max_episodes_per_dataset: Optional[int] = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    seed: int = 42
):
    """
    Convenient function to create a DataLoader for OpenX processed datasets.
    
    Args:
        data_dir: Root directory containing processed data
        split: 'train' or 'val'
        seq_len: Number of frames per sequence
        batch_size: Batch size
        transform: Optional transform to apply to images
        frame_skip: Frame skip for temporal downsampling
        dataset_names: List of specific datasets to load
        max_episodes_per_dataset: Maximum episodes per dataset
        num_workers: DataLoader workers
        pin_memory: Pin memory
        shuffle: Shuffle data
        seed: Random seed
        
    Returns:
        DataLoader
    """
    from torch.utils.data import DataLoader
    
    dataset = OpenXProcessedDataset(
        data_dir=data_dir,
        split=split,
        seq_len=seq_len,
        transform=transform,
        frame_skip=frame_skip,
        dataset_names=dataset_names,
        max_episodes_per_dataset=max_episodes_per_dataset,
        seed=seed
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if split == 'train' else False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    return dataloader
