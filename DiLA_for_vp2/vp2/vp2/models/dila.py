import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import sys
import os
import numpy as np

from .....models.model import STTransformer, Separate_Fusion, Inverse_World_model
from .....models.RAE.rae import RAE
from .....models.modules.action_decoder import ActionMLP

class DiLAModel(nn.Module):
    def __init__(self, 
                 world_model_ckpt_path, 
                 la_enc_ckpt_path=None,
                 n_context=None, 
                 planning_horizon=None,
                 model_depth=16,
                 action_dim=4,
                 epoch=None,
                 n_past=2,
                 device='cuda'):
        super().__init__()
        self.device = device
        
        self.action_dim = action_dim
        self.rae, self.world_model, self.latent_action_encoder = self._init_components(
            model_depth, world_model_ckpt_path, la_enc_ckpt_path
        )
        self.n_context = n_context
        self.num_context = n_past
        self.planning_horizon = planning_horizon
        self.base_prediction_modality = "rgb"
        self.to(device)
        self.eval()
        self.requires_grad_(False)

    def _init_components(self, depth, model_ckpt, la_enc_ckpt):
        # --- RAE Initialization ---
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        dinov2_root = (
            "path/to/dinov2" # replace with actual path to Dinov2 model weights
        )

        rae = RAE(
        encoder_cls="Dinov2withNorm",
        encoder_config_path=dinov2_root,
        encoder_input_size=224,
        encoder_params={"dinov2_path": dinov2_root,
                        "normalize": True},
        decoder_config_path="path/to/decoder/config",  # replace with actual path to decoder config
        decoder_patch_size=16,
        pretrained_decoder_path="path/to/pretrained/decoder",  # replace with actual path to pretrained decoder
        reshape_to_2d=True,
        noise_tau=0.0,
        normalization_stat_path="path/to/normalization/stat",  # replace with actual path to normalization stat
            )
        rae.eval()
        rae.to(self.device)
        for p in rae.parameters(): p.requires_grad_(False)
        print(f'RAE initialized.')

        data = torch.load(model_ckpt, map_location="cpu")
        # Handle potential mismatch if model was saved directly vs. via get_state_dict
        if "model" in data:
            model_state = data["model"]
        elif "module" in data:  # If it was a DDP model state_dict
            model_state = data["module"]
        else:
            model_state = data  # Assume entire data is model state if 'model' key is missing

        # --- Latent Action Encoder Initialization ---
        latent_action_dim = 64
        latent_action_encoder = ActionMLP(num_actions=self.action_dim, action_dim=latent_action_dim)

        # --- World Model Initialization ---
        embedding_dim = 768
        structure_dim = 128
        content_dim = 256

        structure_encoder = STTransformer(embedding_dim).to(self.device)
        content_fusion = Separate_Fusion(embedding_dim, structure_dim, content_dim).to(self.device)

        model = Inverse_World_model(structure_encoder, content_fusion, None).to(self.device)
        if la_enc_ckpt == 'None':
            model.Action_decoder = latent_action_encoder
        model.load_state_dict(model_state, strict=True)

        if la_enc_ckpt != 'None':
            checkpoint = torch.load(la_enc_ckpt, map_location='cpu')
            latent_action_encoder.load_state_dict(checkpoint["model_state_dict"], strict=True)
            latent_action_encoder.eval()
            print(f'Latent Action Encoder Initialized.')
            model.Action_decoder = latent_action_encoder

        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        print(f'World Model Initialized.')

        return rae, model, latent_action_encoder
    
    def batch_resize(self, video_tensor, size):
        """
        Args:
            video_tensor: (B, T, C, H, W)
            size: (new_H, new_W) or int
        Returns:
            resized_tensor: (B, T, C, new_H, new_W)
        """
        x_flat = rearrange(video_tensor, 'b t c h w -> (b t) c h w')
        x_resized = F.interpolate(x_flat, size=size, mode='bilinear', align_corners=False)
        
        b = video_tensor.shape[0]
        x_final = rearrange(x_resized, '(b t) c h w -> b t c h w', b=b)
        
        return x_final
    
    def encode(self, X):
        B, T, _, _, _ = X.size()

        # Encode video frames to latents
        X_enc = rearrange(X, 'b t h w c -> b t c h w')  # (B, T, C, H, W)
        X_enc = self.rae.encode(X_enc.flatten(0,1))
        X_enc = rearrange(X_enc, '(b t) ... -> b t ...', b=B, t=T)

        return X_enc

    @torch.no_grad()
    def __call__(self, batch, n_context=None, planning_horizon=None, grad_enabled=False):
        """
        Input: {'video': (B, T, C, H, W)∈[0, 1], 'actions': (B, T, A)∈[-1, 1]}
        Output: {'rgb': (B, T+n_pred, C, H, W)∈[0, 1]}

        [DEBUG] batch keys: ['video', 'actions', 'state_obs']
        [DEBUG] batch['video'] shape: (200, 2, 64, 64, 3)
        [DEBUG] batch['actions'] shape: (200, 11, 5)
        [DEBUG] base_preds shape: (200, 11, 3, 64, 64)
        [DEBUG] preds keys after base: ['rgb']
        [DEBUG] preds[rgb] shape: (200, 11, 64, 64, 3)

        """
        if n_context is None:
            n_context = self.n_context
        if planning_horizon is None:
            planning_horizon = self.planning_horizon
        batch_size_limit = 200
        total_batch_size = batch['video'].shape[0]
        result_list = []

        with torch.set_grad_enabled(grad_enabled):

            for i in range(0, total_batch_size, batch_size_limit):
                data = {'video': batch['video'][i : i + batch_size_limit],
                        'actions': batch['actions'][i : i + batch_size_limit]}
                if isinstance(data['video'], np.ndarray):
                    X = torch.from_numpy(data['video']).float().to(self.device)
                else:
                    X = data['video']

                if isinstance(data['actions'], np.ndarray):
                    actions = torch.from_numpy(data['actions'][:,:,:]).float().to(self.device)
                else:
                    actions = data['actions'][:,:,:]

                B, T, _, _, _ = X.size()
                X_enc = self.encode(X)

                # Encode actions to latent actions
                actions = rearrange(actions, 'b t a -> (b t) a')

                latent_actions = self.world_model.Action_decoder(actions)
                latent_actions = rearrange(latent_actions, '(b t) d -> b t d', b=B)

                # Make predictions using the world model
                res_dict = self.world_model.autoregressive_forward(X_enc, latent_actions)
                # return {embedding_gen, structure_gen, content_mem}
                g_preds = res_dict['structure_gen']

                preds = res_dict['embedding_gen']
                rec_imgs = self.rae.decode(preds.flatten(0,1)).clamp(0, 1)
                rec_imgs = rearrange(rec_imgs, '(b t) c h w -> b t c h w', b=B)
                rec_imgs = self.batch_resize(rec_imgs, size=(64, 64))
                rec_imgs = rearrange(rec_imgs, 'b t c h w -> b t h w c')  # (B, T, H, W, C)

                result_list.append(rec_imgs if grad_enabled else rec_imgs.detach().cpu().numpy())

            if result_list:
                final_rgb = torch.stack(result_list, dim=0)[0] if grad_enabled else np.concatenate(result_list, axis=0)

        return {'rgb': final_rgb}
