import os
import json
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, 
                 root_dir, 
                 seq_len=10, 
                 stride=1, 
                 transform=None, 
                 split_ratio=0.8,  # Used only when the dataset has no explicit split directories.
                 mode='train',    # 'train', 'test', 'val', 'all'
                 seed=42,
                 return_goal=False,
                 cache_metadata=True,
                 return_state=False):
        
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.stride = stride
        self.transform = transform
        self.split_ratio = split_ratio
        self.mode = mode.lower()
        self.seed = seed
        self.return_goal = return_goal
        self.cache_metadata = cache_metadata
        self.return_state = return_state

        # --- Action Offset ---
        if "memory-maze" in self.root_dir.lower():
            self.action_offset = 1  
            # print(f"[Dataset] Detected 'memory-maze', setting action_offset={self.action_offset}")
        else:
            self.action_offset = 0
            # print(f"[Dataset] Standard dataset, setting action_offset={self.action_offset}")
        
        # Treat validation as test for split selection.
        if self.mode == 'val':
            self.mode = 'test'
        
        self.samples = [] 
        self.json_cache = {} 
        self.split_info = {}
        
        # print(f"[Dataset] Mode: {self.mode.upper()} | Root: {root_dir}")
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
            # is_train_dir = 'train' in path_parts_lower
            # is_test_dir = 'test' in path_parts_lower or 'val' in path_parts_lower or 'validation' in path_parts_lower or 'eval' in path_parts_lower
            is_train_dir = any('train' in p or 'training' in p for p in path_parts_lower)
            is_test_dir = any(
                'test' in p or 'testing' in p or 'val' in p or 'validation' in p or 'eval' in p
                for p in path_parts_lower
            )
            
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
        
        # Save split diagnostics for downstream inspection.
        self.split_info = {
            'total_folders': len(final_folders),
            'source_breakdown': source_stats,
            'ambiguous_pool_size': len(ambiguous),
            'explicit_pool_size': len(explicit_train) + len(explicit_test)
        }
        
        # Debug split information.
        # print(f"  [Split Info] Total: {len(final_folders)}")
        # print(f"  [Split Info] From Explicit Directories: {source_stats['explicit']}")
        # print(f"  [Split Info] From Random Split: {source_stats['random']}")
        
        return final_folders

    def _build_index(self):
        target_folders = self._get_sequence_folders()
        
        # print(f"  - Scanning {len(target_folders)} folders for sequences...")
        
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
        
        # print(f"[Dataset] Initialized. Total samples: {len(self.samples)}")

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
        
        seq_meta = full_meta[start_idx : start_idx + self.seq_len + 1]
        
        # Collect metadata.
        actions, goals = [], []

        if self.return_state:
            # Keep common state fields when requested.
            xs, ys, zs, yaws, pitches = [], [], [], [], []
        
        for frame_data in seq_meta:
            if self.return_state:
                xs.append(frame_data.get('x', 0.0))
                ys.append(frame_data.get('y', 0.0))
                zs.append(frame_data.get('z', 0.0))
                yaws.append(frame_data.get('yaw', 0.0))
                pitches.append(frame_data.get('pitch', 0.0))
            
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

        act_start_idx = start_idx + self.action_offset
        act_end_idx = start_idx + self.seq_len 
        
        action_slice = full_meta[act_start_idx : act_end_idx]
        for frame_data in action_slice:
            actions.append(self._process_action(frame_data.get('action', [])))

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
            'path': item['rel_path']
        }
        if self.return_state:
            result['state'] = torch.tensor([xs, ys, zs, yaws, pitches], dtype=torch.float32).T
        if self.return_goal and len(goals) > 0:
            result['goal'] = torch.stack(goals)

        return result
