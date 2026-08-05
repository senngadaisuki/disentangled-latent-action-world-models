import torch
import torch.nn as nn
from einops import rearrange

from dila_model.models.modules.mamba_ssm import MambaStack


class ContentMemory(nn.Module):
    """
    SSM-based (Mamba) content memory for hierarchical planning.

    This module acts as a stateful memory that:
    1. Accumulates content information across frames during inference
    2. Can be updated autoregressively during generation
    3. Filters relevant semantic content while ignoring motion/structure changes
    
    Architecture:
    - Ingests per-frame content tokens of shape [b, t, h, w, d_c]
    - Applies Mamba SSM with selective state space mechanism
    - Processes each spatial position independently across time
    - Returns memory-refined content per frame, same shape as input
    
    Advantages over Transformer:
    - Linear time complexity vs quadratic for attention
    - Maintains explicit state for autoregressive generation
    - Better at filtering and selective memorization
    - More efficient for long sequences
    """

    def __init__(
        self,
        content_dim_per_patch: int,
        patch_size: int = 4,
        depth: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.dim_per_patch = content_dim_per_patch
        self.depth = depth
        self.n_spatial_patches = patch_size * patch_size

        # Mamba SSM stack for temporal processing
        # Each spatial position has independent temporal dynamics
        self.mamba_stack = MambaStack(
            d_model=self.dim_per_patch,
            n_layers=depth,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
        )
        
        # Optional: learnable aggregation to enhance content representation
        self.content_enhance = nn.Sequential(
            nn.LayerNorm(self.dim_per_patch),
            nn.Linear(self.dim_per_patch, self.dim_per_patch),
            nn.GELU(),
            nn.Linear(self.dim_per_patch, self.dim_per_patch),
        )

    def forward(self, content_tokens: torch.Tensor, states=None) -> torch.Tensor:
        """
        Forward pass for training or batch inference.
        
        Args:
            content_tokens: [b, t, h, w, d_c] - per-frame content features
            states: Optional list of states for each layer (for autoregressive generation)
        
        Returns:
            output: [b, t, h, w, d_c] - memory-refined content
            new_states: Updated states for next autoregressive step
        """
        b, t, h, w, d = content_tokens.shape
        
        # Flatten spatial dimensions
        x = rearrange(content_tokens, 'b t h w d -> b t (h w) d')
        
        # Process each spatial position independently through Mamba
        # This allows each location to maintain its own content memory
        x = rearrange(x, 'b t n d -> (b n) t d')
        
        # Apply Mamba SSM
        x, new_states = self.mamba_stack(x, states)
        
        # Reshape back
        x = rearrange(x, '(b n) t d -> b t n d', b=b, n=h * w)
        
        # Enhance content representation
        x_enhanced = self.content_enhance(x)
        x = x + x_enhanced  # Residual connection
        
        # Reshape to original spatial structure
        out = rearrange(x, 'b t (h w) d -> b t h w d', h=h, w=w)
        
        return out, new_states
    
    def step(self, content_token: torch.Tensor, states=None):
        """
        Single-step forward for autoregressive generation.
        
        This is the key method for online/autoregressive content memory update.
        During generation, call this for each new frame to update the memory state.
        
        Args:
            content_token: [b, 1, h, w, d_c] - single frame content feature
            states: List of hidden states from previous step
        
        Returns:
            output: [b, 1, h, w, d_c] - memory-refined content for this frame
            new_states: Updated states for next step
        """
        # Same as forward but optimized for single timestep
        return self.forward(content_token, states)
    
    def reset_states(self, batch_size, device):
        """
        Reset/initialize states for new sequence generation.
        
        Args:
            batch_size: Batch size
            device: Device to create states on
        
        Returns:
            states: Initialized states (None for fresh start)
        """
        # Mamba handles None states internally
        return None


