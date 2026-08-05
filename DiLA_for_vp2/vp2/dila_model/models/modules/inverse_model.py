# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from einops import rearrange, repeat
from dila_model.models.modules.attention import ContinuousPositionBias, Transformer


class Inverse_model(nn.Module):
    def __init__(self, input_dim=8192, action_dim=4096, hidden_dim=4096, hidden_depth=2, patch_size=4):
        super().__init__()

        # Initial projection to intermediate dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.patch_size = patch_size
        
        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_depth):
            self.hidden_layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, g_prev, g_next):
        """
        g_prev: [B, T-1, D]
        g_next: [B, T-1, D]
        """
        g_prev = rearrange(g_prev, 'b t h w c -> b t (h w c)', h=self.patch_size, w=self.patch_size)   # [B, T-1, H, W, C]
        g_next = rearrange(g_next, 'b t h w c -> b t (h w c)', h=self.patch_size, w=self.patch_size)   # [B, T-1, H, W, C]

        # Temporal difference
        x = g_next - g_prev  # [B, T-1, D]

        # Initial projection
        h = self.input_proj(x)
        
        # Apply hidden layers with residual connections
        for layer in self.hidden_layers:
            h = h + layer(h)
        
        # Final projection
        action = self.output_proj(h)
        return action
    

class ResBlock3D(nn.Module):
    def __init__(self, channels, kernel_size=(3,3,3), padding=(1,1,1)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8, num_channels=channels),
            nn.SiLU(),
            nn.Conv3d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=8, num_channels=channels)
        )
        self.relu = nn.SiLU()

    def forward(self, x):
        return self.relu(x + self.net(x))


class DeepConvIDM_VQ(nn.Module):
    def __init__(self, g_channels=32, action_dim=256, base_channels=64):
        super().__init__()
        # Stem: lift channel capacity while preserving temporal resolution.
        self.stem = nn.Sequential(
            nn.Conv3d(g_channels, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU()
        )
        
        # 2. Stages (ResNet Style)
        # Stage 1: 16x16, preserve resolution.
        self.stage1 = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels)
        )
        
        # Stage 2: 16x16 -> 8x8.
        self.down1 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage2 = nn.Sequential(
            ResBlock3D(base_channels*2),
            ResBlock3D(base_channels*2) 
        )
        
        # Stage 3: 8x8 -> 4x4.
        self.down2 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage3 = nn.Sequential(
            ResBlock3D(base_channels*4),
            ResBlock3D(base_channels*4)
        )

        self.head = nn.Sequential(
            nn.Linear(base_channels*4, 1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, action_dim)
        )


    def forward(self, g_prev, g_next):
        # Input: [B, T, H, W, C] with H=16, W=16, C=32
        x = rearrange(g_next, 'b t h w c -> b c t h w')
        x = self.stem(x)  # [B, 64, T, 16, 16]
        
        x = self.stage1(x)  # [B, 64, T, 16, 16]
        x = self.down1(x)   # [B, 128, T, 8, 8]
        
        x = self.stage2(x)  # [B, 128, T, 8, 8]
        x = self.down2(x)   # [B, 256, T, 4, 4]
        
        x = self.stage3(x)  # [B, 256, T, 4, 4]
        x = rearrange(x, 'b c t h w -> b t h w c')
        return self.head(x)
        

class DeepConvIDM_KL(nn.Module):
    def __init__(self, g_channels=32, action_dim=256, base_channels=64):
        super().__init__()
        
        # Stem: lift channel capacity while preserving temporal resolution.
        self.stem = nn.Sequential(
            nn.Conv3d(g_channels, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU()
        )
        
        # 2. Stages (ResNet Style)
        # Stage 1: 16x16, preserve resolution.
        self.stage1 = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels)
        )
        
        # Stage 2: 16x16 -> 8x8.
        self.down1 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage2 = nn.Sequential(
            ResBlock3D(base_channels*2),
            ResBlock3D(base_channels*2) 
        )
        
        # Stage 3: 8x8 -> 4x4.
        self.down2 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage3 = nn.Sequential(
            ResBlock3D(base_channels*4),
            ResBlock3D(base_channels*4)
        )
        
        # Stage 4: 4x4 -> 2x2.
        self.down3 = nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage4 = nn.Sequential(
            ResBlock3D(base_channels*8),
        )
        
        # Stage 5: 2x2 -> 1x1 (Bottleneck)
        self.down4 = nn.Conv3d(base_channels*8, base_channels*8, kernel_size=3, stride=(1,2,2), padding=1)
        
        # 3. Head
        # Final feature shape: [B, base_channels*8, T, 1, 1].
        # Flattened shape: [B, T, base_channels*8].
        self.action_mu = nn.Sequential(
            nn.Linear(base_channels*8, 1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, action_dim)
        )

        self.action_logvar = nn.Sequential(
            nn.Linear(base_channels*8, 1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, action_dim)
        )

    def forward(self, g_prev, g_next):
        # Input: [B, T, H, W, C] with H=16, W=16, C=32
        x = rearrange(g_next, 'b t h w c -> b c t h w')
        x = self.stem(x)  # [B, 64, T, 16, 16]
        
        x = self.stage1(x)  # [B, 64, T, 16, 16]
        x = self.down1(x)   # [B, 128, T, 8, 8]
        
        x = self.stage2(x)  # [B, 128, T, 8, 8]
        x = self.down2(x)   # [B, 256, T, 4, 4]
        
        x = self.stage3(x)  # [B, 256, T, 4, 4]
        x = self.down3(x)   # [B, 512, T, 2, 2]
        
        x = self.stage4(x)  # [B, 512, T, 2, 2]
        x = self.down4(x)   # [B, 512, T, 1, 1]
        
        # [B, 512, T, 1, 1] -> [B, T, 512]
        x = rearrange(x, 'b c t h w -> b t (h w c)')
        
        # [B, T, 512] -> [B, T, 256]
        mu = self.action_mu(x)
        logvar = self.action_logvar(x)
        return mu, logvar


class DeepConvIDM(nn.Module):
    def __init__(self, g_channels=32, action_dim=256, base_channels=64):
        super().__init__()
        
        # Stem: lift channel capacity while preserving temporal resolution.
        self.stem = nn.Sequential(
            nn.Conv3d(g_channels, base_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.GroupNorm(num_groups=8, num_channels=base_channels),
            nn.SiLU()
        )
        
        # 2. Stages (ResNet Style)
        # Stage 1: 16x16, preserve resolution.
        self.stage1 = nn.Sequential(
            ResBlock3D(base_channels),
            ResBlock3D(base_channels)
        )
        
        # Stage 2: 16x16 -> 8x8.
        self.down1 = nn.Conv3d(base_channels, base_channels*2, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage2 = nn.Sequential(
            ResBlock3D(base_channels*2),
            ResBlock3D(base_channels*2) 
        )
        
        # Stage 3: 8x8 -> 4x4.
        self.down2 = nn.Conv3d(base_channels*2, base_channels*4, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage3 = nn.Sequential(
            ResBlock3D(base_channels*4),
            ResBlock3D(base_channels*4)
        )
        
        # Stage 4: 4x4 -> 2x2.
        self.down3 = nn.Conv3d(base_channels*4, base_channels*8, kernel_size=3, stride=(1,2,2), padding=1)
        self.stage4 = nn.Sequential(
            ResBlock3D(base_channels*8),
        )
        
        # Stage 5: 2x2 -> 1x1 (Bottleneck)
        self.down4 = nn.Conv3d(base_channels*8, base_channels*8, kernel_size=3, stride=(1,2,2), padding=1)
        
        # 3. Head
        # Final feature shape: [B, base_channels*8, T, 1, 1].
        # Flattened shape: [B, T, base_channels*8].
        self.head = nn.Sequential(
            nn.Linear(base_channels*8, 1024),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, action_dim)
        )

        # self.action_mu = nn.Sequential(
        #     nn.Linear(base_channels*8, 1024),
        #     nn.SiLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, action_dim)
        # )

        # self.action_logvar = nn.Sequential(
        #     nn.Linear(base_channels*8, 1024),
        #     nn.SiLU(),
        #     nn.Dropout(0.1),
        #     nn.Linear(1024, action_dim)
        # )

    def forward(self, g_prev, g_next):
        # Input: [B, T, H, W, C] with H=16, W=16, C=32
        x = rearrange(g_next - g_prev, 'b t h w c -> b c t h w')
        x = self.stem(x)  # [B, 64, T, 16, 16]
        
        x = self.stage1(x)  # [B, 64, T, 16, 16]
        x = self.down1(x)   # [B, 128, T, 8, 8]
        
        x = self.stage2(x)  # [B, 128, T, 8, 8]
        x = self.down2(x)   # [B, 256, T, 4, 4]
        
        x = self.stage3(x)  # [B, 256, T, 4, 4]
        x = self.down3(x)   # [B, 512, T, 2, 2]
        
        x = self.stage4(x)  # [B, 512, T, 2, 2]
        x = self.down4(x)   # [B, 512, T, 1, 1]
        
        # [B, 512, T, 1, 1] -> [B, T, 512]
        x = rearrange(x, 'b c t h w -> b t (h w c)')
        
        # [B, T, 512] -> [B, T, 256]
        return self.head(x)
        # mu = self.action_mu(x)
        # logvar = self.action_logvar(x)
        # return mu, logvar


# class ScaledInverseModel(nn.Module):
#     """
#     A scaled-up inverse model that infers a pure latent action `z`.

#     Upgrades:
#     1.  **Deeper Delta Processing:** A Transformer Encoder is used to process the
#         spatial grid of state changes (delta) before pooling. This allows the
#         model to understand complex relationships within the motion itself.
#     2.  **Wider MLP Head:** The final MLP has more capacity for the final projection.
#     """
#     def __init__(self, input_channels: int, action_dim: int, heads: int = 8, dim_head: int = 64, depth: int = 4):
#         super().__init__()
        
#         self.query_token = nn.Parameter(torch.randn(1, 1, action_dim))
#         inner_dim = dim_head * heads

#         # Initial linear projection to match the transformer's expected dimension
#         self.initial_projection = nn.Linear(input_channels, inner_dim)

#         # 1. --- UPGRADE: Transformer Encoder ---
#         # A stack of transformer layers to deeply process the delta grid.
#         transformer_layer = nn.TransformerEncoderLayer(
#             d_model=inner_dim, 
#             nhead=heads, 
#             dim_feedforward=inner_dim * 4,
#             dropout=0.1,
#             activation='gelu',
#             batch_first=True
#         )
#         self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=depth)

#         # Layer to project the transformer's output into Key/Value for attention pooling
#         self.to_kv = nn.Linear(inner_dim, inner_dim * 2, bias=False)

#         self.attention = nn.MultiheadAttention(
#             embed_dim=action_dim, 
#             num_heads=heads, 
#             kdim=inner_dim, 
#             vdim=inner_dim, 
#             batch_first=True
#         )
        
#         # 2. --- UPGRADE: Wider MLP Head ---
#         self.mlp_head = nn.Sequential(
#             nn.LayerNorm(action_dim),
#             nn.Linear(action_dim, action_dim * 4), # Increased width
#             nn.GELU(),
#             nn.Linear(action_dim * 4, action_dim)
#         )

#     def forward(self, g_prev: torch.Tensor, g_next: torch.Tensor) -> torch.Tensor:
#         b, t, h, w, c = g_prev.shape
#         delta = g_next - g_prev
#         delta_flat = rearrange(delta, 'b t h w c -> (b t) (h w) c')

#         # 1. Project and process with Transformer
#         projected_delta = self.initial_projection(delta_flat)
#         processed_delta = self.transformer_encoder(projected_delta)

#         # 2. Project to Key/Value for pooling
#         key, value = self.to_kv(processed_delta).chunk(2, dim=-1)

#         # 3. Attention Pooling (same as before)
#         query = repeat(self.query_token, '1 1 d -> bt 1 d', bt = b * t)
#         pooled_output, _ = self.attention(query, key, value)

#         # 4. Final Processing
#         latent_action_flat = self.mlp_head(pooled_output.squeeze(1))
#         latent_action = rearrange(latent_action_flat, '(b t) d -> b t d', b=b)

#         return latent_action

# class ScaledInverseModel(nn.Module):
#     def __init__(self, 
#                  input_dim=8192, 
#                  hidden_dim=4096,
#                  action_dim=4096, 
#                  patch_size=4,
#                  dim_head=64, 
#                  heads=8,
#                  attn_dropout=0.,
#                  ff_dropout=0.1,
#                  peg=True,
#                  peg_causal=True,
#                  spatial_depth=1,
#                  temporal_depth=1,
#                  hidden_depth=2,
#                  ):
#         super().__init__()
#         self.action_dim = action_dim

#         # Initial projection to intermediate dimension
#         self.input_proj = nn.Linear(input_dim, hidden_dim)
        
#         # Spatial transformer
#         enc_spatial_transformer_kwargs = dict(
#             dim = hidden_dim,
#             dim_head = dim_head,
#             heads = heads,
#             attn_dropout = attn_dropout,
#             ff_dropout = ff_dropout,
#             peg = peg,
#             peg_causal = peg_causal,
#         )

#         # Temporal transformer
#         enc_temporal_transformer_kwargs = dict(
#             dim = hidden_dim,
#             dim_head = dim_head,
#             heads = heads,
#             attn_dropout = attn_dropout,
#             ff_dropout = ff_dropout,
#             peg = peg,
#             peg_causal = peg_causal,
#             causal = True,
#         )

#         # Position bias
#         self.spatial_rel_pos_bias = ContinuousPositionBias(dim = hidden_dim, heads = heads)
#         self.enc_spatial_transformer = Transformer(depth = spatial_depth, **enc_spatial_transformer_kwargs)
#         self.enc_temporal_transformer = Transformer(depth = temporal_depth, **enc_temporal_transformer_kwargs)

#         self.query_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
#         self.pooling_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=8, batch_first=True)
        
#         # Output projection
#         self.output_proj = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.SiLU(),
#             nn.Linear(hidden_dim, action_dim)
#         )

#         # Split the output layer into mu and logvar.
#         # self.fc_mu = nn.Sequential(
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.SiLU(),
#         #     nn.Linear(hidden_dim, action_dim)
#         # )
#         # self.fc_logvar = nn.Sequential(
#         #     nn.LayerNorm(hidden_dim),
#         #     nn.SiLU(),
#         #     nn.Linear(hidden_dim, action_dim)
#         # )

#     def forward(self, g_prev, g_next):
#         # Temporal difference
#         x = g_next - g_prev  # [B, T-1, H, W, D]
#         b, t, h, w, d = x.shape

#         x = rearrange(x, 'b t h w d -> (b t h w) d')
#         x = self.input_proj(x)

#         tokens = rearrange(x, '(b t h w) d -> (b t) (h w) d', b=b, t=t, h=h, w=w)
#         # Process with spatial transformer
#         attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
#         tokens = self.enc_spatial_transformer(
#             tokens, 
#             attn_bias=attn_bias, 
#             video_shape=(b, t, h, w)
#         )
#         tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, t=t, h=h, w=w)

#         # Process with temporal transformer
#         tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
#         tokens = self.enc_temporal_transformer(tokens, video_shape=(b, t, h, w))
#         tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, t=t, h=h, w=w)

#         # tokens: [b, t, h, w, d] -> [b*t, h*w, d]
#         flat_tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

#         # Create query for each timestep
#         query = repeat(self.query_token, '1 1 d -> b 1 d', b=b*t)

#         # Attend: Query looks at all spatial tokens to extract "Action"
#         # resulting shape: [b*t, 1, d]
#         action_token, _ = self.pooling_attention(query, flat_tokens, flat_tokens)
#         action_token = action_token.squeeze(1) # [b*t, d]

#         # Predict mu/logvar from this summary token
#         # mu = self.fc_mu(action_token)
#         # logvar = self.fc_logvar(action_token)

#         # return rearrange(mu, '(b t) d -> b t d', b=b), rearrange(logvar, '(b t) d -> b t d', b=b)

#         action = self.output_proj(action_token)
#         return rearrange(action, '(b t) d -> b t d', b=b, t=t)


class VAEInverseModel(nn.Module):
    def __init__(self, 
                 input_dim=8192, 
                 hidden_dim=4096,
                 action_dim=4096, 
                #  patch_size=4,
                #  dim_head=64, 
                #  heads=8,
                #  attn_dropout=0.,
                #  ff_dropout=0.1,
                #  peg=True,
                #  peg_causal=True,
                #  spatial_depth=1,
                #  temporal_depth=1,
                 hidden_depth=2,
                 ):
        super().__init__()
        self.action_dim = action_dim

        # Initial projection to intermediate dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers with skip connections
        self.hidden_layers = nn.ModuleList()
        for i in range(hidden_depth):
            self.hidden_layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim)
            ))

        # enc_spatial_transformer_kwargs = dict(
        #     dim = input_dim // (patch_size**2),
        #     dim_head = dim_head,
        #     heads = heads,
        #     attn_dropout = attn_dropout,
        #     ff_dropout = ff_dropout,
        #     peg = peg,
        #     peg_causal = peg_causal,
        # )

        # enc_temporal_transformer_kwargs = dict(
        #     dim = input_dim // (patch_size ** 2),
        #     dim_head = dim_head,
        #     heads = heads,
        #     attn_dropout = attn_dropout,
        #     ff_dropout = ff_dropout,
        #     peg = peg,
        #     peg_causal = peg_causal,
        #     causal = True,
        # )

        # self.spatial_rel_pos_bias = ContinuousPositionBias(dim = input_dim, heads = heads)
        # self.enc_spatial_transformer = Transformer(depth = spatial_depth, **enc_spatial_transformer_kwargs)
        # self.enc_temporal_transformer = Transformer(depth = temporal_depth, **enc_temporal_transformer_kwargs)

        # Split the output layer into mu and logvar.
        self.fc_mu = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.fc_logvar = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, g_prev, g_next):
        # g_prev = g_prev.unsqueeze(2)
        # g_next = g_next.unsqueeze(2)
        # x = torch.cat([g_prev, g_next], dim=2)
        # b, t, f, h, w, d = x.shape
        # x = rearrange(x, 'b t f h w c -> (b t) f h w c', f=f, h=h, w=w)   # [B*(T-1), 2, H, W, C]

        # tokens = rearrange(x, 'b t h w d -> (b t) (h w) d')
        # # Process with spatial transformer
        # attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        # tokens = self.enc_spatial_transformer(
        #     tokens, 
        #     attn_bias=attn_bias, 
        #     video_shape=(b*t, f, h, w)
        # )
        # tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b*t, t=f, h=h, w=w)

        # # Process with temporal transformer
        # tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        # tokens = self.enc_temporal_transformer(tokens, video_shape=(b*t, f, h, w))
        # tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b*t, t=f, h=h, w=w)

        # tokens = rearrange(tokens, '(b t) f h w d -> b t f (h w d)', b=b, t=t, f=f, h=h, w=w, d=d)[:, :, 1:].squeeze(2)  # [B, T-1, D]

        # # actions = self.output_proj(tokens)

        # # return actions

        # mu = self.fc_mu(tokens)
        # logvar = self.fc_logvar(tokens)

        # return mu, logvar

        g_prev = rearrange(g_prev, 'b t h w c -> b t (h w c)', h=4, w=4)   # [B, T-1, H, W, C]
        g_next = rearrange(g_next, 'b t h w c -> b t (h w c)', h=4, w=4)   # [B, T-1, H, W, C]

        # Temporal difference
        x = g_next - g_prev  # [B, T-1, D]

        # Initial projection
        h = self.input_proj(x)
        
        # Apply hidden layers with residual connections
        for layer in self.hidden_layers:
            h = h + layer(h)

        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar

# ... imports ...

class ScaledInverseModel(nn.Module):
    def __init__(self, 
                 input_dim=8192, 
                 hidden_dim=4096,
                 action_dim=4096, 
                 patch_size=4,
                 num_latent_tokens=8,
                 dim_head=64, 
                 heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.1,
                 peg=True,
                 peg_causal=True,
                 spatial_depth=2,    # M causal layers
                 temporal_depth=2,   # M causal layers
                 hidden_depth=2,
                 ):
        super().__init__()
        self.action_dim = action_dim
        self.patch_size = patch_size
        self.num_latent_tokens = num_latent_tokens

        # Initial projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer Args
        # transformer_kwargs = dict(
        #     dim = hidden_dim,
        #     dim_head = dim_head,
        #     heads = heads,
        #     attn_dropout = attn_dropout,
        #     ff_dropout = ff_dropout,
        #     peg = peg,
        #     peg_causal = peg_causal,
        # )

        # M causal attention layers (Spatial + Temporal)
        # self.spatial_rel_pos_bias = ContinuousPositionBias(dim = hidden_dim, heads = heads)
        # self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        # self.enc_temporal_transformer = Transformer(depth = temporal_depth, causal=True, **transformer_kwargs)

        # N cross-attention layers (learned query attending to context)
        self.query_token = nn.Parameter(torch.randn(1, self.num_latent_tokens, hidden_dim))
        
        # "N cross-attention layers that transform a learned query token"
        # We use 2 layers here as an example of N
        self.action_transformer = Transformer(
            dim=hidden_dim,
            depth=8, 
            dim_head=dim_head,
            heads=heads,
            has_cross_attn=True,
            # This transformer will do Self-Attn on query (trivial for 1 token)
            # AND Cross-Attn to the context
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, g_prev, g_next):
        # g_prev, g_next: [B, T, H, W, C]
        
        # 1. Concatenate time dimension: [B, 2*T, H, W, C]
        # We process pairs (t, t+1) independently or as a full sequence. 
        # For simplicity matching the description "xt and xt+1", let's stack them.
        # x = torch.stack([g_prev, g_next], dim=2) # [B, T, 2, H, W, C]
        # b, t, f, h, w, c = x.shape
        
        # # Flatten for processing: Treat each (t, t+1) pair as a short video of length 2
        # # Or process the whole batch * time together
        # x = rearrange(x, 'b t f h w c -> (b t) f h w c')
        
        # x_spatial_flat = rearrange(x, 'bt f h w c -> (bt f h w) c')
        # tokens = self.input_proj(x_spatial_flat)
        # tokens = rearrange(tokens, '(bt f h w) d -> (bt f) (h w) d', bt=b*t, f=f, h=h, w=w)

        # # 2. M Causal Attention Layers (Contextualization)
        # # Spatial
        # attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        # tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=(b*t, f, h, w))
        
        # # Temporal (Causal)
        # tokens = rearrange(tokens, '(bt f) (h w) d -> bt f h w d', bt=b*t, f=f, h=h, w=w)
        # tokens = rearrange(tokens, 'bt f h w d -> (bt h w) f d')
        # tokens = self.enc_temporal_transformer(tokens, video_shape=(b*t, f, h, w))
        
        # # Context is now prepared: [bt, f, h, w, d]
        # context = rearrange(tokens, '(bt h w) f d -> bt (f h w) d', bt=b*t, h=h, w=w)

        # Temporal difference
        delta = g_next - g_prev  # [B, T-1, H, W, D]
        b, t, h, w, d = delta.shape

        delta = self.input_proj(rearrange(delta, 'b t h w c -> (b t h w) c'))
        context = rearrange(delta, '(b t h w) c -> (b t) (h w) c', b=b, t=t, h=h, w=w)
        
        # 3. N Cross-Attention Layers (Query -> Action)
        # Query: [bt, 1, d]
        query = repeat(self.query_token, '1 n d -> bt n d', bt=b*t, n=self.num_latent_tokens)
        
        # Transformer with Cross-Attention: Query attends to Context
        # The Transformer class handles: x=query, context=context
        z_tokens = self.action_transformer(query, context=context) # [bt, n, d]
        
        # Project to action
        action = self.output_proj(z_tokens)
        action = rearrange(action, '(b t) n d -> b t (n d)', b=b, t=t, n=self.num_latent_tokens)

        return action
        
