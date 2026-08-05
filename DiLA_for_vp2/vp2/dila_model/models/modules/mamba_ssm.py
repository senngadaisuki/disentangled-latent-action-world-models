"""
Mamba-based State Space Model for Content Memory

This implements a selective state space model (SSM) inspired by Mamba,
designed for autoregressive content memory in hierarchical planning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math


class MambaBlock(nn.Module):
    """
    Single Mamba block with selective state space mechanism.
    
    The selective SSM allows the model to filter and remember relevant content
    information across time, making it ideal for content memory that needs to:
    1. Accumulate semantic information over frames
    2. Filter out irrelevant motion/structure changes
    3. Update efficiently in autoregressive generation
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        dropout: float = 0.0,
        bias: bool = False,
        conv_bias: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)
        else:
            self.dt_rank = dt_rank
        
        # Input projection: x -> (z, x, B, C, dt)
        # z: gating, x: input to SSM, B/C: SSM parameters, dt: step size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        
        # Convolutional layer for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        
        # SSM parameter projections
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        
        # dt projection (step size)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        # Initialize dt projection
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias for stability
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        
        # SSM state matrices A and D
        # A: state transition matrix (fixed structure, parameterized)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            'n -> d n',
            d=self.d_inner,
        )
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        
    def forward(self, x, state=None):
        """
        x: (B, L, D) where B=batch, L=sequence length, D=d_model
        state: Optional hidden state from previous step (for autoregressive generation)
        
        Returns:
            output: (B, L, D)
            new_state: Updated state for next step
        """
        batch, seqlen, dim = x.shape
        
        # Input projection
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)  # Each (B, L, d_inner)
        
        # Causal convolution
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[..., :seqlen]  # Causal: trim to original length
        x = rearrange(x, 'b d l -> b l d')
        
        # Activation
        x = F.silu(x)
        
        # SSM forward
        y, state = self.ssm(x, state)
        
        # Gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        output = self.dropout(output)
        
        return output, state
    
    def ssm(self, x, state=None):
        """
        Selective State Space Model
        
        x: (B, L, D_inner)
        state: (B, D_inner, D_state) or None
        """
        batch, seqlen, d_inner = x.shape
        
        # Get SSM parameters from input
        x_dbl = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        )
        
        # dt: (B, L, dt_rank) -> (B, L, d_inner)
        dt = self.dt_proj(dt)
        dt = F.softplus(dt)  # Ensure positive
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Discretize continuous parameters
        # Using zero-order hold (ZOH) discretization
        dA = torch.exp(torch.einsum('bld,dn->bldn', dt, A))  # (B, L, d_inner, d_state)
        dB = torch.einsum('bld,bln->bldn', dt, B)  # (B, L, d_inner, d_state)
        
        # SSM recurrence
        if state is None:
            state = torch.zeros(batch, d_inner, self.d_state, device=x.device, dtype=x.dtype)
        
        # Autoregressive state update
        outputs = []
        for t in range(seqlen):
            # Update state: h_t = A * h_{t-1} + B * x_t
            state = dA[:, t] * state + dB[:, t] * x[:, t:t+1, :].transpose(1, 2)
            
            # Output: y_t = C * h_t + D * x_t
            y = torch.einsum('bdn,bn->bd', state, C[:, t])
            y = y + self.D * x[:, t]
            
            outputs.append(y)
        
        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        
        return y, state


class MambaLayer(nn.Module):
    """
    Mamba layer with pre-norm and residual connection.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            **kwargs
        )
    
    def forward(self, x, state=None):
        """
        x: (B, L, D)
        state: Optional hidden state
        """
        residual = x
        x = self.norm(x)
        x, new_state = self.mamba(x, state)
        x = x + residual
        return x, new_state


class MambaStack(nn.Module):
    """
    Stack of Mamba layers for deep state space modeling.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x, states=None):
        """
        x: (B, L, D)
        states: List of states for each layer, or None
        
        Returns:
            output: (B, L, D)
            new_states: List of updated states
        """
        if states is None:
            states = [None] * len(self.layers)
        
        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            new_states.append(new_state)
        
        x = self.norm(x)
        return x, new_states

