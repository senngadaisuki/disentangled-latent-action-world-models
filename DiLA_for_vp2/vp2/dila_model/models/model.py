import torch
import torch.nn as nn
import einops

from dila_model.models.modules.inverse_model import DeepConvIDM
from dila_model.models.modules.forward_model import AdaForwardDynamics
from dila_model.models.modules.st_transformer import STTransformerBackbone
from dila_model.models.modules.fusion_decoder import CANNEncoder, CANNDecoder, FusionDecoder, StructureEncoderSpatialTemporal
from dila_model.models.modules.content_memory import ContentMemory

from dila_model.utils.losses import mse_loss, covariance_reg_loss, var_loss, action_loss, motion_consistency_loss, \
    time_variance_loss, symmetry_loss, l2_loss, kl_divergence_loss

from einops import rearrange
from einops.layers.torch import Rearrange

class STTransformer(nn.Module):
    def __init__(self, 
                 Embedding_dim,
                 channels=768,
                 H=16,
                 W=16,
                 patch_size=1,
                 spatial_depth=4,
                 temporal_depth=4,
                 dim_head=64,
                 heads=8,
                 gamma=0.98,
                 epsilon=1e-8,
                 num_negatives=3,
                 batch_first=True,
                 sensory_inf_loss=10.,
                 vip_inf_loss=0.1,
                 covariance_loss=0.001,
                 variance_loss=0.001
                ):
        super(STTransformer, self).__init__()
        self.channels = channels
        self.H = H
        self.W = W
        self.patch_size = patch_size
        patch_height = self.H // self.patch_size
        patch_width = self.W // self.patch_size
        self.embedding_dim = Embedding_dim

        self.gamma = gamma
        self.epsilon = epsilon
        self.num_negatives = num_negatives

        self.sensory_inf_loss = sensory_inf_loss
        self.vip_inf_loss = vip_inf_loss
        self.covariance_loss = covariance_loss
        self.variance_loss = variance_loss

        st_transformer_configs = dict(
            embedding_dim = self.embedding_dim,
            patch_size = patch_size,
            dim_head = dim_head,
            heads = heads,
            spatial_depth = spatial_depth,
            temporal_depth = temporal_depth,
            causal = True
        )
        self.ST_model = STTransformerBackbone(**st_transformer_configs)

        self.batch_first = batch_first

    def forward(self, x):
        embedding_features = self.ST_model(x)

        return {
            'embedding_features': embedding_features,
        }

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # Orthogonal initialization for weights
                elif 'bias' in name:
                    nn.init.zeros_(param)  # Initialize bias to 0
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def to(self, device):
        super().to(device)
        self.ST_model.to(device)
        return self
    
    def configure_optimizers(self, weight_decay, lr, betas, T_max):
        # Configure main optimizer with AdamW
        world_optimizer = torch.optim.AdamW(self.parameters(), lr=lr['world_lr'], betas=betas, weight_decay=weight_decay)
        world_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(world_optimizer, T_max=T_max, eta_min=1e-5)

        return world_optimizer, world_scheduler


class Separate_Fusion(nn.Module):
    def __init__(self, 
                 Embedding_dim,
                 Structure_dim,
                 content_dim,
                 patch_size=1,  
                 spatial_depth=3,
                 temporal_depth=4,
                 dim_head=32,
                 heads=8,
                 ssm_depth=2,
                 ssm_d_state=32,
                 ssm_d_conv=4,
                 ssm_expand=2,
                 ssm_dropout=0.0,
                 batch_first=True,
               ):
        super(Separate_Fusion, self).__init__()
        self.patch_size = 16 // patch_size
        self.embedding_dim = Embedding_dim
        self.structure_dim = Structure_dim
        self.content_dim = content_dim  # 4 times larger than grid_dim

        fusion_decoder_configs = dict(
            embedding_dim = self.embedding_dim,
            patch_size = patch_size,
            dim_head = dim_head,
            heads = heads,
            spatial_depth = spatial_depth,
            peg=True,
            peg_causal=True,
            # for ablation: g only
            dim_context = self.content_dim,
            has_cross_attn = True,  
        )

        self.fusion_decoder = FusionDecoder(**fusion_decoder_configs)

        # for ablation: g only, no content
        # self.fusion_decoder = StructureEncoderSpatialTemporal(**fusion_decoder_configs)

        # Embedding -> structure
        self.motion_encoder = CANNEncoder(
            input_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.structure_dim,
            hidden_depth=2
        )

        # Embedding -> content
        self.content_encoder = CANNEncoder(
            input_dim=self.embedding_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.content_dim,
            hidden_depth=2
        )

        # Content memory to aggregate content features causally over time
        self.content_memory = ContentMemory(
            content_dim_per_patch=self.content_dim,
            patch_size=self.patch_size,
            depth=ssm_depth,
            d_state=ssm_d_state,
            d_conv=ssm_d_conv,
            expand=ssm_expand,
            dropout=ssm_dropout,
        )

        self.structure_to_intermediate = CANNDecoder(
            n_factors=self.structure_dim,
            hidden_dim=self.embedding_dim,
            output_dim=self.embedding_dim,
            hidden_depth=2
        )

        # Initialize batch_first flag
        self.batch_first = batch_first

    def fuse_and_decode(self, g, content_frame, content_memory):
        g = self.structure_to_intermediate(g)
        p = self.fusion_decoder(g, context=(content_frame, content_memory))
        return p

    # for ablation: g only, no content
    # def fuse_and_decode(self, g):
    #     g = self.structure_to_intermediate(g)
    #     p = self.fusion_decoder(g)
    #     return p
    
    def forward(self, embedding_features, content_states=None):
        # Structure encoding (motion/position)
        structure_inf = self.motion_encoder(embedding_features)

        # Content: per-frame encoding then memory aggregation via Mamba
        content_frame = self.content_encoder(embedding_features)               # [b, t, h, w, d_c]
        content_mem, new_content_states = self.content_memory(
            content_frame,
            states=content_states
        )  # [b, t, h, w, d_c], updated states
        
        return {
            'structure_inf': structure_inf,
            'content_frame': content_frame,
            'content_mem': content_mem,
            'content_states': new_content_states,
        }

     # Calculate total loss as weighted sum of individual losses
    def total_loss(self, res):
        embedding_features = rearrange(res['embedding_features'], 'b t h w d -> b t (h w d)')
        embedding_int = rearrange(res['embedding_int'], 'b t h w d -> b t (h w d)')
        loss_embedding_features_mse = mse_loss(embedding_features, embedding_int)
    
        # structure_inf = torch.atan2(res['structure_inf'][..., 1], res['structure_inf'][..., 0])  # [B, T, N, 2] -> [B, T, N]
        structure_inf = rearrange(res['structure_inf'], 'b t h w d -> b t (h w d)')
        loss_covariance = covariance_reg_loss(structure_inf)
        loss_var = var_loss(rearrange(structure_inf, 'b t d -> b (t d)'))
        loss_time_var = time_variance_loss(structure_inf, gamma=0.1)

        losses_dict = {
            'place_inf_mse': loss_embedding_features_mse,
            'covariance': loss_covariance,
            'variance': loss_var,
            'temporal_variance': loss_time_var,
        }

        weights = {
            'place_inf_mse': 1.,  # 1
            'covariance': 0.05,  # 0.05
            'variance': 0.05,
            'temporal_variance': 0.05,
        }

        # Calculate total weighted loss
        total_loss = sum([weights[k] * v for k, v in losses_dict.items()])
        return total_loss, losses_dict
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.ConvTranspose2d):
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')  # Kaiming initialization
            if module.bias is not None:
                nn.init.zeros_(module.bias)  # Initialize bias to 0
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.orthogonal_(param)  # Orthogonal initialization for weights
                elif 'bias' in name:
                    nn.init.zeros_(param)  # Initialize bias to 0
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def to(self, device):
        super().to(device)
        self.motion_encoder.to(device)
        self.content_memory.to(device)
        self.structure_to_intermediate.to(device)
        self.fusion_decoder.to(device)
        return self
    
    def configure_optimizers(self, weight_decay, lr, betas, T_max):
        # Configure main optimizer with AdamW
        world_optimizer = torch.optim.AdamW(self.parameters(), lr=lr['world_lr'], betas=betas, weight_decay=weight_decay)
        world_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(world_optimizer, T_max=T_max, eta_min=1e-5)

        return world_optimizer, world_scheduler


class Inverse_World_model(nn.Module):
    def __init__(self, 
                 structure_encoder,
                 content_fusion,
                 Action_decoder=None,
                 Embedding_dim=512,
                 Structure_dim=128,
                 Action_dim=64,
                 patch_size=16,
                 dim_head=32,
                 heads=8,
                 hidden_dim=256,
                 depth=4,
                 batch_first=True,
                 phase=1,
                 sensory_inf_loss=1,
                 sensory_int_loss=1,
                 sensory_gen_loss=1, 
                 place_gen_loss=1, 
                 place_int_loss=1,
                 grid_loss=1,
                 action_loss=1, 
                 content_consistency_loss=1,
                 mutual_info_loss=1,
                 covariance_loss=1,
                 variance_loss=1,
                 kl_action_loss=1,
                 symmetry_loss=1,
                 l2_loss=1,
                 ):
        super(Inverse_World_model, self).__init__()

        self.structure_encoder = structure_encoder
        self.content_fusion = content_fusion
        self.Action_decoder = Action_decoder

        self.patch_size = patch_size
        self.action_dim = Action_dim
        self.structure_dim = Structure_dim 
        self.embedding_dim = Embedding_dim
        self.phase = phase

        self.sensory_inf_loss = sensory_inf_loss
        self.sensory_gen_loss = sensory_gen_loss
        self.sensory_int_loss = sensory_int_loss
        self.place_gen_loss = place_gen_loss
        self.place_int_loss = place_int_loss
        self.grid_loss = grid_loss
        self.action_loss = action_loss
        self.content_consistency_loss = content_consistency_loss
        self.mutual_info_loss = mutual_info_loss
        self.covariance_loss = covariance_loss
        self.variance_loss = variance_loss
        self.kl_action_loss = kl_action_loss
        self.l2_loss = l2_loss
        self.symmetry_loss = symmetry_loss

        self.Inverse_model = DeepConvIDM(
            g_channels=self.structure_dim,
            action_dim=self.action_dim,
            base_channels=self.structure_dim*2
        )

        self.forward_dynamics = AdaForwardDynamics(
            g_dim=self.structure_dim, 
            z_dim=self.action_dim, 
            hidden_dim=hidden_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=0.1,
            ff_dropout=0.1
        )

        # Initialize batch_first flag
        self.batch_first = batch_first

    def forward(self, x, actions=None):
        # -----phase 1-----
        if self.phase == 1:
            res = self.one_step_forward(x)
            loss, loss_dict = self.total_loss(res)

        # -----phase 2-----
        elif self.phase == 2:
            latent_actions, structure_inf = self.get_latent_actions(x, return_structure_inf=True)
            res_dict = self.autoregressive_forward(x[:, :1], latent_actions)
            res_dict = {**res_dict, 'x': x, 'structure_inf': structure_inf, 'latent_actions': latent_actions}
            loss, loss_dict = self.total_loss(res_dict)

        # -----phase 3-----
        elif self.phase == 3:
            if actions is None:
                raise ValueError("Phase 3 requires ground-truth actions.")
            if self.Action_decoder is None:
                raise ValueError("Phase 3 requires an Action_decoder.")

            actions = actions[:, 1:-1]
            batch_size, seq_len, *_ = actions.size()
            actions = actions.flatten(0, 1)
            z = self.Action_decoder(actions)
            z = rearrange(z, '(b t) d -> b t d', b=batch_size, t=seq_len)
            gt_z, structure_inf = self.get_latent_actions(x[:, 1:], return_structure_inf=True)
            res_dict = self.prediction(x[:, :2], z)
            res_dict = {**res_dict, 'x': x[:, 1:], 'gt_z': gt_z, 'structure_inf': structure_inf, 'z': z, 'latent_actions': z}
            loss, loss_dict = self.total_loss(res_dict)

        return loss, loss_dict
    
    def one_step_forward(self, x):
        batch_size, seq_len, *_ = x.size()

        encoder_result = self.structure_encoder(rearrange(x, 'b t c h w -> b t h w c'))
        embedding_features = encoder_result['embedding_features']
        
        fusion_result = self.content_fusion(embedding_features, content_states=None)
        structure_inf = fusion_result['structure_inf']
        content_mem = fusion_result['content_mem']

        structure_inf_prev = structure_inf[:, :-1]
        structure_inf_next = structure_inf[:, 1:]
        latent_actions = self.Inverse_model(structure_inf_prev, structure_inf_next)

        # # Generate in parallel: predict delta_structure and add to current state
        delta_structure = self.forward_dynamics(structure_inf[:, :-1], latent_actions)    # Predict change in grid cells
        structure_gen = structure_inf[:, :-1] + delta_structure

        # Construct content sequence for parallel decoding
        x_start = rearrange(x[:, 0:1].expand(-1, seq_len - 1, -1, -1, -1), 'b t c h w -> b t h w c') # [B, T-1, ...]

        embedding_gen = self.content_fusion.fuse_and_decode(structure_gen, x_start, content_mem[:, :-1])
        embedding_gen = rearrange(embedding_gen, 'b t h w c -> b t c h w')

        res = {
            'x': x,
            'embedding_gen': embedding_gen,
            'structure_inf': structure_inf,
            'structure_gen': structure_gen,
            'latent_actions': latent_actions,
            'content_mem': content_mem,
        }
        return res

    def transfer(self, x_a, x):
        batch_size, seq_len, *_ = x_a.size()

        fusion_result_a = self.content_fusion(self.structure_encoder(rearrange(x_a, 'b t c h w -> b t h w c'))['embedding_features'], content_states=None)
        structure_inf_a = fusion_result_a['structure_inf']

        # structure_inf_prev = torch.zeros_like(structure_inf_a[:, :-1])
        structure_inf_prev = structure_inf_a[:, :-1]
        structure_inf_next = structure_inf_a[:, 1:]
        latent_actions = self.Inverse_model(structure_inf_prev, structure_inf_next)

        encoder_result = self.structure_encoder(rearrange(x, 'b t c h w -> b t h w c'))
        embedding_features = encoder_result['embedding_features']
        fusion_result = self.content_fusion(embedding_features, content_states=None)
        structure_inf = fusion_result['structure_inf']
        content_mem = fusion_result['content_mem']

        delta_structure = self.forward_dynamics(structure_inf[:, :-1], latent_actions)    # Predict change in grid cells
        structure_gen = structure_inf[:, :-1] + delta_structure
        
        # structure_gen = self.forward_dynamics(structure_inf[:, :-1], latent_actions)

        x_start = rearrange(x[:, 0:1].expand(-1, seq_len - 1, -1, -1, -1), 'b t c h w -> b t h w c') # [B, T-1, ...]

        # content_mem_start = content_mem[:, 0:1].expand(-1, seq_len - 1, -1, -1, -1) # [B, T-1, ...]
        embedding_gen = self.content_fusion.fuse_and_decode(structure_gen, x_start, content_mem[:, :-1])

        # s_pred = self.content_fusion.fuse_and_decode(structure_inf[:, 1:], x_start, content_mem[:, :-1])

        return {
            'x': x,
            'embedding_gen': embedding_gen,
            # 's_pred': s_pred,
        }

    def get_latent_actions(self, x, return_structure_inf=False):
        encoder_result = self.structure_encoder(rearrange(x, 'b t c h w -> b t h w c'))
        embedding_features = encoder_result['embedding_features']
        fusion_result = self.content_fusion(embedding_features, content_states=None)
        structure_inf = fusion_result['structure_inf']
        # structure_inf_prev = torch.zeros_like(structure_inf[:, :-1])
        structure_inf_prev = structure_inf[:, :-1]
        structure_inf_next = structure_inf[:, 1:]
        latent_actions = self.Inverse_model(structure_inf_prev, structure_inf_next) # [B, T_frame-1, Action_dim]

        # latent_actions_mu, latent_actions_logvar = self.Inverse_model(structure_inf_prev, structure_inf_next)
        # latent_actions = latent_actions_mu + torch.randn_like(latent_actions_mu) * torch.exp(latent_actions_logvar * 0.5)

        if return_structure_inf:
            return latent_actions, structure_inf
        else:
            return latent_actions

        # if return_structure_inf:
            # return latent_actions, latent_actions_mu, latent_actions_logvar, structure_inf
        # else:
            # return latent_actions

    def prediction(self, x_context, latent_actions):
        t_pred = latent_actions.size(1)

        encoder_result_context = self.structure_encoder(rearrange(x_context, 'b t c h w -> b t h w c'))
        embedding_features_context = encoder_result_context['embedding_features']
        fusion_result_context = self.content_fusion(embedding_features_context, content_states=None)
        structure_inf_context = fusion_result_context['structure_inf']
        content_mem_context = fusion_result_context['content_mem']
        content_states = fusion_result_context['content_states']

        x_start = rearrange(x_context[:, :1], 'b t c h w -> b t h w c')
        # history_g = structure_inf_context
        current_structure = structure_inf_context[:, -1:]
        next_content_mem = content_mem_context[:, -1:]

        embedding_gen_list = []
        structure_gen_list = []
        for i in range(t_pred):
            current_action = latent_actions[:, i:i+1]  # [B, 1, action_dim]
            delta_structure = self.forward_dynamics(current_structure, current_action)
            next_structure_gen = current_structure + delta_structure
            current_structure = next_structure_gen

            next_embedding_gen = self.content_fusion.fuse_and_decode(next_structure_gen, x_start, next_content_mem)
            
            next_p = self.structure_encoder(next_embedding_gen)['embedding_features']
            next_content = self.content_fusion.content_encoder(next_p)
            next_content_mem, content_states = self.content_fusion.content_memory.step(next_content, states=content_states)
            embedding_gen_list.append(next_embedding_gen)
            structure_gen_list.append(next_structure_gen)

        embedding_gen = torch.cat(embedding_gen_list, dim=1)
        structure_gen = torch.cat(structure_gen_list, dim=1)
        embedding_gen = rearrange(embedding_gen, 'b t h w c -> b t c h w')
        return {
            'embedding_gen': embedding_gen,
            'structure_gen': structure_gen,
        }

    def autoregressive_forward(self, x_context, latent_actions):
        t_pred = latent_actions.size(1) - x_context.size(1) + 1
        t_context = x_context.size(1)

        encoder_result_context = self.structure_encoder(rearrange(x_context, 'b t c h w -> b t h w c'))
        embedding_features_context = encoder_result_context['embedding_features']
        fusion_result_context = self.content_fusion(embedding_features_context, content_states=None)
        structure_inf_context = fusion_result_context['structure_inf']
        content_mem_context = fusion_result_context['content_mem']
        content_states = fusion_result_context['content_states']

        x_start = rearrange(x_context[:, :1], 'b t c h w -> b t h w c')
        current_structure = structure_inf_context[:, 0:1]
        next_content_mem = content_mem_context[:, 0:1]

        embedding_gen_list = []
        structure_gen_list = []
        content_mem_list = []
        for i in range(t_context+t_pred-1):
            current_action = latent_actions[:, i:i+1]  # [B, 1, action_dim]
            delta_structure = self.forward_dynamics(current_structure, current_action)
            next_structure_gen = current_structure + delta_structure
            # next_structure_gen = self.forward_dynamics(current_structure, current_action)

            next_embedding_gen = self.content_fusion.fuse_and_decode(next_structure_gen, x_start, next_content_mem)
            # next_embedding_gen = self.content_fusion.fuse_and_decode(next_structure_gen)

            if i < t_context-1:
                next_content_mem = content_mem_context[:, i+1:i+2]
                current_structure = structure_inf_context[:, i+1:i+2]
                # current_structure = next_structure_gen
            else:
                current_structure = next_structure_gen
                next_p = self.structure_encoder(next_embedding_gen)['embedding_features']
                next_content = self.content_fusion.content_encoder(next_p)
                # next_content_mem = next_content
                next_content_mem, content_states = self.content_fusion.content_memory.step(next_content, states=content_states)
                content_mem_list.append(next_content_mem)
            embedding_gen_list.append(next_embedding_gen)
            structure_gen_list.append(next_structure_gen)

        embedding_gen = torch.cat(embedding_gen_list, dim=1)
        structure_gen = torch.cat(structure_gen_list, dim=1)
        content_mem = torch.cat(content_mem_list, dim=1)
        embedding_gen = rearrange(embedding_gen, 'b t h w c -> b t c h w')
        return {
            'embedding_gen': embedding_gen,
            'structure_gen': structure_gen,
            'content_mem': content_mem,
        }
    
    # Calculate total loss as weighted sum of individual losses
    def total_loss(self, res):
        loss_embedding_gen = mse_loss(res['x'][:, 1:], res['embedding_gen'])
        # loss_s_pred = mse_loss(res['x'][:, 1:], res['s_pred'])

        # Compare predicted delta_structure with true delta_structure
        loss_structure_inf_gen = action_loss(res['structure_inf'], res['latent_actions'], self.forward_dynamics)
        # loss_structure_inf_gen = mse_loss(res['structure_inf'][:, 1:], res['structure_gen'])
        # 4. Action Loss
        loss_action = motion_consistency_loss(res['structure_inf'], res['structure_gen'], self.Inverse_model)
        # loss_action = mse_loss(res['z'], res['gt_z'])

        g_prev = res['structure_inf'][:, :-1].detach()
        g_next = res['structure_inf'][:, 1:].detach()
        loss_symmetry = symmetry_loss(g_prev, g_next, self.Inverse_model)

        loss_l2 = l2_loss(res['latent_actions'])

        losses_dict = {
            'sensory_gen': loss_embedding_gen,
            'grid': loss_structure_inf_gen,
            'action': loss_action,
            'l2': loss_l2,
            'symmetry': loss_symmetry,
        }

        # Focus more on action learning initially
        weights = {
            'sensory_gen': self.sensory_gen_loss, #3.,
            'grid': self.grid_loss, #5.,
            'action': self.action_loss, #1.,  # Increase action loss importance
            'l2': self.l2_loss,
            'symmetry': self.symmetry_loss,
        }

        # Calculate total weighted loss
        total_loss = sum([weights[k] * v for k, v in losses_dict.items()])
        return total_loss, losses_dict
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        num_params = sum(p.numel() for p in self.parameters())
        return num_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Xavier/Glorot initialization for better gradient flow
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Conv2d):
            # Kaiming initialization matched to activation function
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.ConvTranspose2d):
            # Kaiming initialization for transposed convolutions
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            # For Mamba's causal convolutions
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    # Input-hidden weights: Xavier initialization
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    # Hidden-hidden weights: Orthogonal initialization
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    # Initialize forget gate bias to 1 for better gradient flow
                    nn.init.zeros_(param)
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)  # Forget gate bias = 1
        elif isinstance(module, nn.LayerNorm):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # Initialize attention weights
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                nn.init.xavier_uniform_(module.in_proj_weight)
            if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                nn.init.zeros_(module.in_proj_bias)
            if hasattr(module, 'out_proj'):
                nn.init.xavier_uniform_(module.out_proj.weight)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)
        

    def to(self, device):
        super().to(device)
        self.Inverse_model.to(device)
        self.forward_dynamics.to(device)
        self.structure_encoder.to(device)
        self.content_fusion.to(device)
        return self

    def configure_optimizers(self, weight_decay, lr, betas, T_max):
        # Ensure all relevant parameters require gradients initially
        for param in self.parameters(): # Or be more specific if needed
            param.requires_grad = True
        
        # for param in self.structure_encoder.parameters():
        #     param.requires_grad = False
        # for param in self.content_fusion.parameters():
        #     param.requires_grad = False

        # for param in self.Inverse_model.parameters():
        #     param.requires_grad = False

        # Define parameter groups
        # new_params = list(self.forward_dynamics.parameters()) + list(self.Action_decoder.parameters())
        new_params = list(self.Inverse_model.parameters()) + list(self.forward_dynamics.parameters())
        fusion_params = list(self.content_fusion.parameters())
        encoder_params = list(self.structure_encoder.parameters())

        params_to_train = [
            {'params': new_params, 'lr': lr['inverse_lr'] * 1},     # Higher for inverse model
            {'params': fusion_params, 'lr': lr['world_lr'] * 1},
            {'params': encoder_params, 'lr': lr['world_lr'] * 1}
        ]

        # Configure optimizer with AdamW
        optimizer = torch.optim.AdamW(
            params_to_train,
            betas=betas,
            weight_decay=weight_decay
        )

        # Learning rate scheduler (will scale group LRs proportionally)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=1e-6 # Set eta_min relative to the lowest LR group
        )

        return optimizer, scheduler
