import os
import numpy as np
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import json
import re

from typing import Any, Callable, Optional, Tuple, Union

from torch import stack
import torchvision.transforms.functional as F
import einops
from pathlib import Path
from utils.core import TrajectoryDataset

def exists(val: Any) -> bool:
    return val is not None


def default(val: Optional[Any], d: Any) -> Any:
    return val if exists(val) else d


def pair(val: Any) -> Tuple[Any, Any]:
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


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


def calculate_perplexity(indices, codebook_size):
    """Calculates the perplexity of codebook usage."""
    if indices is None or indices.numel() == 0:
        return torch.tensor(0.0)
    
    # Ensure indices are flat and on CPU for bincount if needed, though bincount works on GPU
    indices = indices.flatten()
    
    # Count occurrences of each index
    counts = torch.bincount(indices, minlength=codebook_size)
    
    # Calculate probability distribution
    probs = counts.float() / counts.sum()
    
    # Calculate entropy H = -sum(p * log2(p))
    # Add epsilon for numerical stability (log2(0) is -inf)
    entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
    
    # Perplexity = 2^H
    perplexity = torch.pow(2, entropy)
    
    return perplexity

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
            raise ValueError("Cannot satisfy the sampling constraints; reduce num_points or min_distance.")
        
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
            clips = [clip[i:i+self.sliding_window] for i in range(self.frames_per_clip - self.sliding_window + 1)]

        return torch.stack(clips, dim=0)  # [5, 4, C, H, W]

        # return torch.stack(clip, dim=0)


class SimKitchenTrajectoryDataset(TrajectoryDataset):
    def __init__(self, data_directory, prefetch=True, onehot_goals=False):
        self.data_directory = Path(data_directory)
        states = torch.from_numpy(np.load(self.data_directory / "observations_seq.npy"))
        actions = torch.from_numpy(np.load(self.data_directory / "actions_seq.npy"))
        goals = torch.load(self.data_directory / "onehot_goals.pth")
        # The current values are in shape T x N x Dim, move to N x T x Dim
        self.states, self.actions, self.goals = transpose_batch_timestep(
            states, actions, goals
        )
        self.Ts = np.load(self.data_directory / "existence_mask.npy").sum(axis=0).astype(int).tolist()
        
        self.prefetch = prefetch
        if self.prefetch:
            self.obses = []
            for i in range(len(self.Ts)):
                self.obses.append(torch.load(self.data_directory / "obses" / f"{i:03d}.pth"))
        self.onehot_goals = onehot_goals

    def get_seq_length(self, idx):
        return self.Ts[idx]

    def get_all_actions(self):
        result = []
        # mask out invalid actions
        for i in range(len(self.Ts)):
            T = self.Ts[i]
            result.append(self.actions[i, :T, :])
        return torch.cat(result, dim=0)

    def get_frames(self, idx, frames):
        # obs, act, mask / obs, act, mask, goal
        if self.prefetch:
            obs = self.obses[idx][frames]
        else:
            obs = torch.load(self.data_directory / "obses" / f"{idx:03d}.pth")[frames]
        obs = obs / 255.0
        act = self.actions[idx, frames]
        mask = torch.ones((len(frames)))
        if self.onehot_goals:
            goal = self.goals[idx, frames]
            return obs, act, mask, goal
        else:
            return obs, act, mask

    def __getitem__(self, idx):
        T = self.Ts[idx]
        return self.get_frames(idx, range(T))
    
    def __len__(self):
        return len(self.Ts)



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

# Example usage
if __name__ == "__main__":
    from torchvision import transforms

    root_dir = "./data/ssv2/rawframes"
    train_file = "./data/ssv2/labels/train.json"
    val_file = "./data/ssv2/labels/validation.json"
    test_file = "./data/ssv2/labels/test.json"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SomethingSomethingV2Dataset(root_dir=root_dir, annotations_file=train_file, transform=transform)
    val_dataset = SomethingSomethingV2Dataset(root_dir=root_dir, annotations_file=val_file, transform=transform)
    test_dataset = SomethingSomethingV2Dataset(root_dir=root_dir, annotations_file=test_file, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    print(len(train_dataset), len(val_dataset), len(test_dataset))
    print(len(train_loader), len(val_loader), len(test_loader))

    # for i, (clips) in enumerate(train_loader):
    #     print(f"Batch {i+1}: {clips.size()}")
