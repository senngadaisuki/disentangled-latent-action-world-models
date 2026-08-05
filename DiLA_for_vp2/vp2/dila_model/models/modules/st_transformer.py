import torch
import torch.nn as nn
import numpy as np
from dila_model.models.modules.attention import ContinuousPositionBias, Transformer
from einops import rearrange


class PlaceCANN1D(nn.Module):
    def __init__(self, neuron_num=100, k_mec=4., z_min=-torch.pi, z_max=torch.pi,
               tau=1., a_g=0.5, J0=4., dt=0.1, batch_first=True):
        super(PlaceCANN1D, self).__init__()

        # parameters
        self.tau = tau
        self.k = k_mec  # Global inhibition
        self.a = a_g  # Range of excitatory connections
        self.J0 = J0  # maximum connection value
        self.neuron_num = neuron_num
        self.z_min = z_min
        self.z_max = z_max
        self.z_range = z_max - z_min
        self.dt = dt

        # variables
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x = torch.tensor(np.linspace(self.z_min, self.z_max, self.neuron_num, endpoint=False), device=self.device)  # The encoded feature values
        self.r = torch.zeros(neuron_num, device=self.device)
        self.u = torch.zeros(neuron_num, device=self.device)
        self.input = torch.zeros(neuron_num, device=self.device)
        self.center = torch.zeros(1, device=self.device)
        self.center_input = torch.zeros(1, device=self.device)
        # Connections
        self.conn_mat = self.make_conn()
        self.conn_fft = torch.fft.fft(self.conn_mat)

        self.batch_first = batch_first

    def period_bound(self, A):
        d = torch.remainder(A, self.z_range)
        d = torch.where(d > 0.5 * self.z_range, d - self.z_range, d)
        return d

    def make_conn(self):
        d = self.period_bound(torch.abs(self.x[0] - self.x))
        Jxx = self.J0 / (self.a * torch.sqrt(torch.tensor(2 * torch.pi))) * torch.exp(-0.5 * torch.square(d / self.a))
        return Jxx

    def get_input_pos(self, pos, input_strength=1.):
        assert pos.dim() == 2 and pos.size(1) == 1, "pos shape should be (batch_size, 1)"
        batch_size = pos.size(0)
        d = self.period_bound(torch.abs(self.x.unsqueeze(0).expand(batch_size, -1) - pos))
        input = input_strength * torch.exp(-0.25 * torch.square(d / self.a))
        return input

    def reset_state(self):
        self.r = torch.zeros(self.neuron_num, device=self.device)
        self.u = torch.zeros(self.neuron_num, device=self.device)
        self.v = torch.zeros(self.neuron_num, device=self.device)
        self.input = torch.zeros(self.neuron_num, device=self.device)

    def get_center(self):
        exppos = torch.exp(1j * self.x)
        self.center = torch.angle(torch.sum(exppos * self.r, dim=-1)).float()
        self.center_input = torch.angle(torch.sum(exppos * self.input, dim=-1))

    def integration(self, x):
        # Calculate recurrent input
        r_fft = torch.fft.fft(self.r, dim=-1)  # Specify the dimension for FFT
        Irec = torch.real(torch.fft.ifft(r_fft * self.conn_fft))  # Unsqueeze conn_fft to match batch size
        # input
        self.input = x
        # Update neural state
        du = (-self.u + self.input + Irec) / self.tau * self.dt
        u = self.u + du
        self.u = torch.where(u > 0, u, 0)
        r1 = torch.square(self.u)
        r2 = 1.0 + self.k * torch.sum(r1, dim=-1)
        self.r = r1 / r2.unsqueeze(-1)
        # Calculate self position
        self.get_center()
        return self.u.float()
    
    def forward(self, x, u0=None, last_time=100, blank_time=100, keep_trace=False):
        if self.batch_first:
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        if u0 is not None:
            # x: [Batch, time, feature]
            assert u0.dim() == 2 and u0.size(1) == self.neuron_num, f"u0 shape should be (batch_size, {self.neuron_num})"
            self.u = u0
            self.r = torch.square(self.u) / (1.0 + self.k * torch.sum(torch.square(self.u), dim=-1)).unsqueeze(-1)
        us = [] if keep_trace else None
        for i in range(seq_len):
            for _ in range(last_time):
                u = self.integration(x[i])
                if keep_trace:
                    us.append(u)

            for _ in range(blank_time):
                u = self.integration(0)
                if keep_trace:
                    us.append(u)
        
        if keep_trace:
            us = torch.stack(us)
            if self.batch_first:
                us = us.transpose(0, 1)

        return self.u.float(), self.r.float(), us.float() if keep_trace else None

    def to(self, device):
        self.device = device
        self.x = self.x.to(device)
        self.r = self.r.to(device)
        self.u = self.u.to(device)
        self.input = self.input.to(device)
        self.center = self.center.to(device)
        self.center_input = self.center_input.to(device)

        self.conn_fft = self.conn_fft.to(device)
        return super().to(device)


class STTransformerBackbone(nn.Module):
    def __init__(self,
                 embedding_dim, 
                 patch_size=4, 
                 dim_head=64, 
                 heads=16, 
                 attn_dropout=0.,
                 ff_dropout=0.1,
                 peg=True,
                 peg_causal=True,
                 spatial_depth=1, 
                 temporal_depth=1,
                 causal=True):
        super(STTransformerBackbone, self).__init__()
        enc_spatial_transformer_kwargs = dict(
            dim = embedding_dim // (patch_size ** 2),
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = peg,
            peg_causal = peg_causal,
            peg_spatial_or_temporal = 'spatial',
        )

        enc_temporal_transformer_kwargs = dict(
            dim = embedding_dim // (patch_size ** 2),
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = peg,
            peg_causal = peg_causal,
            peg_spatial_or_temporal = 'temporal',
            causal = causal,
        )

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = embedding_dim // (patch_size ** 2), heads = heads)
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **enc_spatial_transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **enc_temporal_transformer_kwargs)
        
    
    def forward(self, x):
        b, t, h, w, d = x.shape
        # Process with spatial transformer
        tokens = rearrange(x, 'b t h w d -> (b t) (h w) d')
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)
        tokens = self.enc_spatial_transformer(
            tokens, 
            attn_bias=attn_bias, 
            video_shape=(b, t, h, w)
        )
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, t=t, h=h, w=w)
        
        # Process with temporal transformer
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')
        tokens = self.enc_temporal_transformer(tokens, video_shape=(b, t, h, w))
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, t=t, h=h, w=w)

        return tokens
    
    def to(self, device):
        self.spatial_rel_pos_bias = self.spatial_rel_pos_bias.to(device)
        self.enc_spatial_transformer = self.enc_spatial_transformer.to(device)
        self.enc_temporal_transformer = self.enc_temporal_transformer.to(device)
        return super().to(device)