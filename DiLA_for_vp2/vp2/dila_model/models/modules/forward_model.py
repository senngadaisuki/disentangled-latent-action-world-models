# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from dila_model.models.modules.attention import STTransformer, ContinuousPositionBias, FeedForward

class MLPResidualBlock(nn.Module):
    def __init__(self, dim, ff_mult=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.ff(self.norm(x))
        return x
    
class GatedResidualForward(nn.Module):
    def __init__(self, g_dim=4096, z_dim=512, hidden_dim=4096, depth=4, patch_size=4):
        super().__init__()
        self.patch_size = patch_size
        self.init_proj = nn.Linear(g_dim + z_dim, g_dim)

        self.blocks = nn.ModuleList([
            MLPResidualBlock(dim=g_dim, ff_mult=4) for _ in range(depth)
        ])

        self.delta_proj = nn.Sequential(
            nn.LayerNorm(g_dim),
            nn.Linear(g_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, g_dim),
        )

    def forward(self, g, z):
        g = rearrange(g, 'b t h w d -> b t (h w d)', h=self.patch_size, w=self.patch_size)
        x = torch.cat([g, z], dim=-1)
        x = self.init_proj(x)

        for block in self.blocks:
            x = block(x)  # Residual connection

        delta_g = self.delta_proj(x)
        delta_g = rearrange(delta_g, 'b t (h w d) -> b t h w d', h=self.patch_size, w=self.patch_size)
        return delta_g


class AdaForwardDynamics(nn.Module):
    def __init__(self, g_dim=16, z_dim=256, hidden_dim=512, depth=4, dim_head=64, heads=8, attn_dropout=0.1, ff_dropout=0.1):
        super().__init__()
        
        self.g_dim = g_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(g_dim, hidden_dim),
            FeedForward(hidden_dim, mult=4.0, dropout=ff_dropout),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            FeedForward(hidden_dim, mult=4.0, dropout=ff_dropout),
        )

        dec_st_transformer_kwargs = dict(
            dim=hidden_dim,
            dim_cond=hidden_dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            causal=True,
            peg=True,
            peg_causal=True,
            enable_conditioning=True,
        )
        
        self.dec_spatial_rel_pos_bias = ContinuousPositionBias(dim=hidden_dim, heads=heads, num_dims=2)   
        self.transformer = STTransformer(**dec_st_transformer_kwargs)
        # Project transformer output to delta_g
        self.to_delta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, g_dim),
        )

    def forward(self, g, z):
        """
        Predict delta_g using adaptive spatio-temporal transformer.
        
        Args:
            g: [B, T-1, H, W, C] - current grid cell state
            z: [B, T-1, D] - latent action
            
        Returns:
            delta_g: [B, T-1, H, W, C] - predicted change in grid cells
        """
        b, t, h, w, c = g.shape
        
        # Reshape g to [B, T, N, C] where N = H * W
        x = rearrange(g, 'b t h w c -> b t (h w) c')

        x = self.input_proj(x)
        z = self.action_proj(z)

        tokens = torch.cat([x, rearrange(z, "b t d -> b t 1 d")], dim=2)  # (B, T-1, H*W + 1, D)
        attn_bias = self.dec_spatial_rel_pos_bias(h, w, device=tokens.device, dtype=tokens.dtype)  # (h, Np, Np)
        attn_bias = F.pad(attn_bias, (0, 1, 0, 1), value=0.0)  # (h, Np + 1, Np + 1)

        # Apply transformer with conditioning
        # x: [B, T, N, C], cond: [B, T, D]
        x = self.transformer(tokens, video_shape=(b, t, h, w), cond=z, spatial_attn_bias=attn_bias)

        x = x[:, :, :-1, :]
        
        # Predict delta_g
        delta_g = self.to_delta(x)
        
        # Reshape back to spatial format
        delta_g = rearrange(delta_g, 'b t (h w) c -> b t h w c', h=h, w=w)
        
        return delta_g
        

# Helper Module: Residual Convolutional Block
class ResConvBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels) # GroupNorm is often more stable than BatchNorm
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + residual

class ScaledForwardDynamics(nn.Module):
    """
    A scaled-up forward dynamics model that predicts delta_g.
    
    Architecture philosophy:
    - Predicts the CHANGE in grid cells (delta_g) rather than next state directly
    - This provides better inductive bias: g_{t+1} = g_t + f(g_t, z_t)
    - Action z_t modulates how we extract dynamics from current state g_t
    
    Key components:
    1. **Deep State Encoder:** Extracts rich spatial features from g_t
    2. **Cross-Attention or Spatial Modulation:** Action z_t interacts with spatial features
    3. **Delta Predictor:** Projects modulated features to delta_g space
    """
    def __init__(self, g_channels: int, action_dim: int, depth: int = 4, use_attention: bool = True):
        super().__init__()
        
        self.use_attention = use_attention

        # 1. Deep State Encoder: Extract rich features from current state g_t
        self.state_encoder = nn.Sequential(
            nn.Conv2d(g_channels, g_channels*2, kernel_size=1),
            *[ResConvBlock(g_channels*2) for _ in range(depth)]
        )
        
        # 2. Action Encoder: Process action into meaningful representation
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, g_channels*2), # Map to channel dim
            nn.LayerNorm(g_channels*2),
            nn.GELU()
        )

        if self.use_attention:
             # Cross-Attention: Query=Grid, Key=Action, Value=Action
             # This allows each grid cell to attend to the global action
             self.cross_attention = nn.MultiheadAttention(
                 embed_dim=g_channels*2,
                 num_heads=8,
                 batch_first=True
             )
        else:
            # Spatial Modulation (FiLM-like)
            # Action generates scale and shift for features
            self.action_to_film = nn.Sequential(
                nn.Linear(g_channels * 2, g_channels * 2),
                nn.GELU(),
                nn.Linear(g_channels * 2, g_channels * 2)
            )
        
        # 4. Delta Predictor: Project modulated features to delta_g
        # Uses residual blocks for better gradient flow
        self.delta_predictor = nn.Sequential(
            ResConvBlock(g_channels*2),
            ResConvBlock(g_channels*2),
            nn.GroupNorm(8, g_channels*2),
            nn.GELU(),
            nn.Conv2d(g_channels*2, g_channels, kernel_size=1),
        )
        
        # Optional: Small residual scaling factor for stability
        # Helps prevent large changes early in training
        self.delta_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, g: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Predict delta_g based on current state g_t and action z_t.
        
        Args:
            g: [b, t, h, w, c] - current grid cell state
            z: [b, t, d] - latent action
            
        Returns:
            delta_g: [b, t, h, w, c] - predicted change in grid cells
        """
        b, t, h, w, c = g.shape
        
        # Flatten batch and time dimensions for processing
        g_flat = rearrange(g, 'b t h w c -> (b t) c h w')
        z_flat = rearrange(z, 'b t d -> (b t) d')

        # 1. Extract features from current state
        state_features = self.state_encoder(g_flat)  # [(bt), c, h, w]
        
        # 2. Process action
        action_emb = self.action_encoder(z_flat) # [(bt), c]

        if self.use_attention:
             # Prepare for attention:
             # Query: Flattened spatial grid [(bt), (hw), c]
             # Key/Value: Action [(bt), 1, c]
             
             grid_query = rearrange(state_features, 'bt c h w -> bt (h w) c')
             action_kv = rearrange(action_emb, 'bt c -> bt 1 c')
             
             # Cross-attention: Grid attends to Action
             # Since action is 1 token, this is effectively broadcasting with learnable weights
             # but allows extending to multi-token actions later if needed.
             attended_features, _ = self.cross_attention(
                 query=grid_query,
                 key=action_kv,
                 value=action_kv
             )
             
             # Add residual connection to original grid features
             # This allows the grid to maintain its identity while incorporating action info
             modulated_features_flat = grid_query + attended_features
             
             # Reshape back to spatial
             modulated_features = rearrange(modulated_features_flat, 'bt (h w) c -> bt c h w', h=h, w=w)
             
        else:
            # FiLM Modulation
            film_params = self.action_to_film(action_emb)
            gamma, beta = film_params.chunk(2, dim=-1)  # Each: [bt, c]
            
            gamma = rearrange(gamma, 'bt c -> bt c 1 1')
            beta = rearrange(beta, 'bt c -> bt c 1 1')
            
            modulated_features = state_features * (1 + gamma) + beta

        # 5. Predict delta_g from modulated features
        delta_g_flat = self.delta_predictor(modulated_features)
        
        # Optional: Scale the delta for training stability
        delta_g_flat = delta_g_flat * self.delta_scale
        
        # Reshape back to original format
        delta_g = rearrange(delta_g_flat, '(b t) c h w -> b t h w c', b=b)

        return delta_g


# ... imports ...
# from models.modules.attention import Transformer, ContinuousPositionBias

# class TransformerForwardDynamics(nn.Module):
#     def __init__(self, 
#                  g_dim, 
#                  action_dim, 
#                  hidden_dim=1024,
#                  num_latent_tokens=4,
#                  depth=8, 
#                  heads=8, 
#                  dim_head=64,
#                  patch_size=4):
#         super().__init__()
#         self.patch_size = patch_size
#         self.num_latent_tokens = num_latent_tokens
#         # Encoders
#         self.input_proj = nn.Linear(g_dim, hidden_dim)
#         self.action_proj = nn.Linear(action_dim * num_latent_tokens, hidden_dim)
#         self.pos_bias = ContinuousPositionBias(dim=hidden_dim, heads=heads)

#         # "L blocks composed of self-attention on xt... and cross-attention from xt to zt"
#         self.transformer = Transformer(
#             dim=hidden_dim,
#             depth=depth,
#             dim_head=dim_head,
#             heads=heads,
#             has_cross_attn=True, # Enables the cross-attention block
#             peg=True,
#             peg_causal=True
#         )

#         # Prediction Head
#         self.to_delta = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, g_dim)
#         )

#     def forward(self, g, z):
#         # g: [B, T, H, W, C]
#         # z: [B, T, D]
#         b, t, h, w, c = g.shape
        
#         # Flatten to sequence: [BT, HW, C]
#         g_flat = rearrange(g, 'b t h w c -> (b t) (h w) c')
#         z_flat = rearrange(z, 'b t d -> (b t) 1 d') # Action is a single token per timestep
        
#         # Embed
#         x = self.input_proj(g_flat)      # [BT, HW, Dim]
#         context = self.action_proj(z_flat) # [BT, 1, Dim] (Memory for cross-attn)
        
#         # Transformer Decoder Blocks
#         # 1. Self-Attention on x (handled by transformer)
#         # 2. Cross-Attention: x attends to context (z) (handled by transformer)
#         # 3. FeedForward
#         attn_bias = self.pos_bias(h, w, device=x.device)

#         # print(x.shape, context.shape)
#         # x = torch.cat([context, x], dim=1)
        
#         x = self.transformer(
#             x, 
#             context=context, # z is the key/value
#             attn_bias=attn_bias,
#             video_shape=(b, t, h, w)
#         )
        
#         # Remove action tokens, keep only patch predictions
#         # x = x[:, 1:]                 # [BT, HW, D]
        
#         # Predict Delta
#         delta_g = self.to_delta(x)
#         delta_g = rearrange(delta_g, '(b t) (h w) c -> b t h w c', b=b, t=t, h=h, w=w)
        
#         return delta_g
