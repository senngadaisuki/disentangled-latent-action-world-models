import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange

def off_diag(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def off_diag_cov_loss(x: torch.Tensor) -> torch.Tensor:
    cov = torch.cov(einops.rearrange(x, "... E -> E (...)"))
    return off_diag(cov).square().mean()

def wrap_interval(x, lower=-torch.pi, upper=torch.pi):
    L = upper - lower
    return torch.remainder(x - lower, L) + lower

def sim(tensor1, tensor2):
    d = -torch.linalg.norm(tensor1 - tensor2, dim = -1)
    return d

def mse_loss(s_actual, s_pred):
    mse = nn.MSELoss()
    s_actual = s_actual
    s_pred = s_pred
    return mse(s_pred, s_actual)

def cross_entropy_loss(logits_flat, targets_flat):
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1) 
    return loss_fn(logits_flat, targets_flat)

def cosine_similarity_loss(s_actual, s_pred):
    # Calculate cosine similarity
    cos_sim = F.cosine_similarity(s_pred, s_actual, dim=-1)
    # Define loss: ideally, the similarity should be 1, so loss = 1 - cos_sim
    loss = (1 - cos_sim).mean()
    return loss

def motion_consistency_loss(g_inf, g_gen, inverse_dynamics):
    """Ensure g captures motion by enforcing consistency across time"""
    # Motion should be predictable from previous states
    la_true = inverse_dynamics(g_inf[:, :-1], g_inf[:, 1:])
    la_pred = inverse_dynamics(g_inf[:, :-1], g_gen)

    # Add action diversity regularization
    action_std = la_true.std(dim=[0,1]).mean()
    diversity_loss = torch.exp(-action_std * 10)  # Penalize if std is too low
    return F.mse_loss(la_pred, la_true) + 0.5 * diversity_loss

# Add proper inverse model loss:
def inverse_model_loss(g_inf, latent_actions, forward_dynamics):
    """Train inverse model to predict correct actions"""
    g_prev = g_inf[:, :-1]
    g_next_true = g_inf[:, 1:]
    
    # Forward dynamics prediction
    g_next_pred = g_prev + forward_dynamics(g_prev, latent_actions)
    
    return F.mse_loss(g_next_pred, g_next_true)


def content_invariance_loss(content):
    """Content should be temporally stable within sequences"""
    # 1. Temporal smoothness.
    temporal_diff = content[:, 1:] - content[:, :-1]
    temporal_loss = (temporal_diff ** 2).mean()

    # 2. InfoNCE term to avoid collapse and learn discriminative features.
    loss_nce = content_contrastive_loss(content)
    
    return temporal_loss * 0.5 + loss_nce


def content_contrastive_loss(content, temperature=0.07):
    """
    Args:
        content: [B, T, H, W, D] or [B, T, D]
        temperature: Controls distribution sharpness, typically 0.07 or 0.1.
    Returns:
        loss: InfoNCE loss
    """
    B, T, D = content.shape
    
    # Construct positive/negative pairs.
    # For each sequence in the batch, sample two frames.
    # Anchor: t_i, Positive: t_j (i != j)
    # Negative: all frames from other videos in the batch.
    
    # Compute the loss for all sequences in parallel.
    # A stronger variant uses the average feature across all timesteps as anchor.
    
    # Use temporal average pooling as a sequence fingerprint.
    seq_fingerprint = content.mean(dim=1) # [B, D]
    
    # Normalize before cosine-similarity logits.
    seq_fingerprint = F.normalize(seq_fingerprint, dim=1)
    
    # Compute similarity matrix [B, B].
    # logits[i, j] is the similarity between video i and video j.
    logits = torch.matmul(seq_fingerprint, seq_fingerprint.T) / temperature
    
    # Diagonal entries are positives; off-diagonal entries are negatives.
    labels = torch.arange(B, device=content.device)
    
    # Cross entropy over the batch.
    loss = F.cross_entropy(logits, labels)
    
    return loss

# def content_invariance_loss(content_per_frame, content):
#     temporal_diff = content_per_frame[:, 1:] - content_per_frame[:, :-1]
#     temporal_loss = (temporal_diff ** 2).mean()

#     global_var_loss = ((content_per_frame - content) ** 2).mean()
#     return (temporal_loss + global_var_loss) / 2


def mutual_info_loss(g_inf, content, alpha=1.0, beta=0.5, gamma=0.3):
    """Composite disentanglement loss."""
    g_flat = rearrange(g_inf, 'b t d -> (b t) d')
    content_flat = rearrange(content, 'b t d -> (b t) d')
    
    # Linear correlation.
    g_std = (g_flat - g_flat.mean(dim=0, keepdim=True)) / (g_flat.std(dim=0, keepdim=True) + 1e-6)
    content_std = (content_flat - content_flat.mean(dim=0, keepdim=True)) / (content_flat.std(dim=0, keepdim=True) + 1e-6)
    cross_corr = torch.mm(g_std.T, content_std) / g_std.shape[0]
    linear_corr_loss = (cross_corr ** 2).mean()
    
    # Nonlinear correlation after a tanh transform.
    g_nonlinear = torch.tanh(g_std)
    content_nonlinear = torch.tanh(content_std)
    nonlinear_cross_corr = torch.mm(g_nonlinear.T, content_nonlinear) / g_std.shape[0]
    nonlinear_corr_loss = (nonlinear_cross_corr ** 2).mean()
    
    # Higher-order moment correlation.
    g_squared = (g_std ** 2) - 1.0
    content_squared = (content_std ** 2) - 1.0
    squared_cross_corr = torch.mm(g_squared.T, content_squared) / g_std.shape[0]
    higher_order_loss = (squared_cross_corr ** 2).mean()
    
    total_mi_loss = (alpha * linear_corr_loss + 
                     beta * nonlinear_corr_loss + 
                     gamma * higher_order_loss)
    
    return total_mi_loss

def covariance_reg_loss(obs_enc: torch.Tensor):
    return off_diag_cov_loss(obs_enc)

def var_loss(Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    Z = Z - Z.mean(dim=0)
    std_z = torch.sqrt(Z.var(dim=0) + eps)
    return F.relu(gamma - std_z).mean()

def time_variance_loss(Z: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    Z = Z - Z.mean(dim=1, keepdim=True)
    std_z = torch.sqrt(Z.var(dim=1) + eps)
    return F.relu(std_z - gamma).mean()

def smoothing_loss(g_inf):
    """
    Encourage that delta_g is smooth over time.
    """
    delta_g = g_inf[:, 1:] - g_inf[:, :-1]         # (b, t-1, d)
    delta_delta_g = delta_g[:, 1:] - delta_g[:, :-1]   # (b, t-2, d)
    loss = (delta_delta_g**2).mean()
    return loss

def g_alignment_loss(g_inf, g_gen):
    mse = F.mse_loss(g_inf, g_gen)
    # cosine = 1 - F.cosine_similarity(g_inf, g_gen, dim=-1).mean()
    # loss = mse + cosine * 0.5
    loss = mse
    return loss

def action_loss(g_inf, latent_actions, forward_dynamics):
    delta_g_true = g_inf[:, 1:] - g_inf[:, :-1]
    delta_g_pred = forward_dynamics(g_inf[:, :-1], latent_actions)
    return F.mse_loss(delta_g_pred, delta_g_true)

def adversarial_disentanglement_loss(g_inf, content, discriminator, 
                                    adv_weight=1.0, similarity_weight=1.0):
    """
    Adversarial disentanglement loss with detached discriminator inputs.
    
    Args:
        g_inf: grid representations [B, T, grid_dim]
        content: content representations [B, T, content_dim] 
        discriminator: scene discriminator network
        adv_weight: weight for adversarial loss
        similarity_weight: weight for content similarity loss
    
    Returns:
        encoder_loss, discriminator_loss, loss_dict
    """
    batch_size, seq_len, grid_dim = g_inf.shape
    
    # Sample two timesteps from the same video.
    n_share = min(3, seq_len // 3)
    max_offset = seq_len - n_share
    
    if max_offset <= n_share + 1:
        # Sequence is too short for the adversarial term.
        return torch.tensor(0.0, device=g_inf.device), torch.tensor(0.0, device=g_inf.device), {}
    
    # Build batch indices.
    batch_indices = torch.arange(batch_size).unsqueeze(1)  # [B, 1]
    
    # Sample frames for the discriminator.
    random_t1 = torch.randint(0, n_share, (batch_size,))
    random_t2 = torch.randint(0, n_share, (batch_size,))
    
    g1_sample = g_inf[batch_indices.squeeze(), random_t1]  # [B, grid_dim]
    g2_sample = g_inf[batch_indices.squeeze(), random_t2]  # [B, grid_dim]
    
    # Content should remain similar within the same video.
    mean_content = content.mean(dim=1, keepdim=True)
    content_sim_loss = F.mse_loss(content, mean_content.expand_as(content))
    
    # Train discriminator on detached grid representations.
    # Positive pairs: grids from the same video.
    target_positive = torch.ones((batch_size, 1), device=g_inf.device)
    disc_output_pos = discriminator(g1_sample.detach(), g2_sample.detach())
    disc_loss_pos = F.binary_cross_entropy(disc_output_pos, target_positive)
    
    # Negative pairs: grids from different videos.
    perm_indices = torch.randperm(batch_size)
    g1_neg = g1_sample[perm_indices].detach()
    target_negative = torch.zeros((batch_size, 1), device=g_inf.device)
    disc_output_neg = discriminator(g1_sample.detach(), g1_neg)
    disc_loss_neg = F.binary_cross_entropy(disc_output_neg, target_negative)
    
    discriminator_loss = (disc_loss_pos + disc_loss_neg) * 0.5

    # Grid encoder adversarial loss: make same-video grids ambiguous to the discriminator.
    target_confusion = torch.full((batch_size, 1), 0.5, device=g_inf.device)
    disc_output_adv = discriminator(g1_sample, g2_sample)
    grid_adv_loss = F.binary_cross_entropy(disc_output_adv, target_confusion)
    
    # Encoder loss.
    encoder_loss = similarity_weight * content_sim_loss + adv_weight * grid_adv_loss
    
    loss_dict = {
        'content_similarity': content_sim_loss,
        'grid_adversarial': grid_adv_loss,
        'discriminator_pos': disc_loss_pos,
        'discriminator_neg': disc_loss_neg
    }
    
    return encoder_loss, discriminator_loss, loss_dict

def kl_divergence_loss(mu, logvar):
    # KL(q || p) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1)
    return kl_loss.mean()


def temporal_contrastive_loss(
    g: torch.Tensor,
    temperature: float = 0.1,
    time_delta_range: int = 5
) -> torch.Tensor:
    """
    Compute temporal contrastive loss for g representations.
    Representations from the same sequence are encouraged to be more similar
    than representations from different sequences.

    Args:
        g (torch.Tensor): Dynamic state representation with shape (B, T, D).
                          B: batch_size, T: sequence_length, D: feature_dimension.
        temperature (float): Temperature used to scale InfoNCE similarities.
        time_delta_range (int): Minimum timestep gap for positives within the
            same sequence, preventing adjacent-frame shortcuts.

    Returns:
        torch.Tensor: Scalar contrastive loss.
    """
    assert g.dim() == 3, "Input tensor g must have shape (B, T, D)"
    batch_size, seq_len, dim = g.shape
    
    if seq_len <= time_delta_range:
        return torch.tensor(0.0, device=g.device, requires_grad=True)

    # Prepare anchors and positives.
    anchor_time = torch.randint(0, seq_len - time_delta_range, (batch_size,), device=g.device)
    
    positive_time_offset = torch.randint(time_delta_range, seq_len - anchor_time.max(), (batch_size,), device=g.device)
    positive_time = anchor_time + positive_time_offset

    # Gather anchor and positive representations.
    batch_indices = torch.arange(batch_size, device=g.device)
    anchors = g[batch_indices, anchor_time]      # (B, D)
    positives = g[batch_indices, positive_time]  # (B, D)
    
    # Prepare negatives.
    all_samples = einops.rearrange(g, 'b t d -> (b t) d') # (B*T, D)

    # Compute similarities.
    anchors_norm = F.normalize(anchors, p=2, dim=-1)
    all_samples_norm = F.normalize(all_samples, p=2, dim=-1)
    
    # Similarity matrix between anchors and all samples.
    # (B, D) @ (D, B*T) -> (B, B*T)
    sim_matrix = torch.einsum('bd,nd->bn', anchors_norm, all_samples_norm)
    
    # InfoNCE loss.
    logits = sim_matrix / temperature
    
    # The target for each anchor is the corresponding positive sample index.
    positive_indices = batch_indices * seq_len + positive_time
    
    # Cross entropy over all candidate samples.
    # logits: (B, B*T), positive_indices: (B,)
    loss = F.cross_entropy(logits, positive_indices)
    
    return loss

def action_grounded_contrastive_loss(
    g: torch.Tensor,
    inverse_model: torch.nn.Module,
    temperature: float = 0.1,
    similarity_threshold: float = 0.6
) -> torch.Tensor:
    """
    Compute action-grounded contrastive loss.
    Similar state changes (delta g) should produce similar latent actions.

    Args:
        g (torch.Tensor): Dynamic state representation with shape (B, T, D_g).
        inverse_model (torch.nn.Module): Inverse dynamics model.
        temperature (float): InfoNCE temperature.
        similarity_threshold (float): Cosine-similarity threshold for positive
            delta-g pairs.

    Returns:
        torch.Tensor: Scalar contrastive loss.
    """
    assert g.dim() == 3, "Input tensor g must have shape (B, T, D)"
    batch_size, seq_len, g_dim = g.shape

    if seq_len < 2:
        return torch.tensor(0.0, device=g.device, requires_grad=True)

    # Prepare state-transition pairs.
    # g_prevs: (B, T-1, D_g) -> ((B*(T-1)), D_g)
    g_prevs = einops.rearrange(g[:, :-1], 'b t d -> (b t) d')
    # g_nexts: (B, T-1, D_g) -> ((B*(T-1)), D_g)
    g_nexts = einops.rearrange(g[:, 1:], 'b t d -> (b t) d')

    num_transitions = g_prevs.shape[0]

    # Compute the supervision signal (delta g).
    delta_g = g_nexts - g_prevs
    delta_g_norm = F.normalize(delta_g, p=2, dim=-1)
    
    # Pairwise similarity matrix for all delta_g vectors.
    # (N, D_g) @ (D_g, N) -> (N, N), where N = B*(T-1)
    sim_delta_g = torch.einsum('id,jd->ij', delta_g_norm, delta_g_norm)

    # Positive mask: delta_g pairs above the similarity threshold.
    positive_mask = sim_delta_g > similarity_threshold
    # Remove self-pairs.
    positive_mask.fill_diagonal_(False)
    
    # Skip when no positive pairs are available.
    if positive_mask.sum() == 0:
        return torch.tensor(0.0, device=g.device, requires_grad=True)

    # Compute latent action representations.
    a = inverse_model(g_prevs, g_nexts)
    a_norm = F.normalize(a, p=2, dim=-1)

    # Contrastive loss over latent actions.
    # (N, D_a) @ (D_a, N) -> (N, N)
    sim_a = torch.einsum('id,jd->ij', a_norm, a_norm)
    logits = sim_a / temperature

    # Multi-positive contrastive objective.
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    
    mean_log_prob_pos = (log_prob * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
    
    loss = -mean_log_prob_pos.mean()

    return loss

def l1_sparsity_loss(tensor):
    """
    Compute L1 norm loss to encourage sparsity on the feature dimension.
    This promotes disentanglement by encouraging only a few grid cells to be active.
    
    Args:
        tensor: Input tensor of shape [B, T, H, W, D] or [B, T, D]
    
    Returns:
        L1 norm loss (scalar)
    """
    return torch.abs(tensor).mean()

def l2_loss(tensor):
    return torch.norm(tensor, p=2, dim=-1).mean()


def symmetry_loss(g_prev, g_next, inverse_model, threshold=0.01):
    a_forward = inverse_model(g_prev, g_next)
    a_backward = inverse_model(g_next, g_prev)

    delta_g = g_next - g_prev
    motion_magnitude = torch.norm(
        delta_g.reshape(delta_g.shape[0], delta_g.shape[1], -1), 
        dim=-1
    )  # [B, T-1]
    
    # Apply the constraint only when motion is significant.
    motion_mask = (motion_magnitude > threshold).float()

    cos_sim = F.cosine_similarity(a_forward, a_backward, dim=-1)  # [B, T-1]
        
    # Minimize (cos + 1)^2; the ideal cosine similarity is -1.
    asymmetry_loss = (cos_sim + 1.0) ** 2

    weighted_loss = (asymmetry_loss * motion_mask).sum() / (motion_mask.sum() + 1e-8)
    return weighted_loss


def isotropy_loss(g_inf, latent_actions, alpha=0.1):
    # 1. Calculate the L2 norm (squared) of delta_g over the feature dimension
    # Shape: (B, S-1, D_g) -> (B, S-1)
    delta_g = g_inf[:, 1:] - g_inf[:, :-1]
    delta_g_sq_norm = torch.sum(delta_g**2, dim=-1)
    
    # 2. Calculate the "stillness gate" w_t
    # This weight is ~1 when delta_g is ~0, and ~0 otherwise.
    # Shape: (B, S-1)
    with torch.no_grad(): # Gate should not receive gradients
        stillness_gate = torch.exp(-alpha * delta_g_sq_norm)
        
    # 3. Calculate the L1 norm of z over the feature dimension
    # Shape: (B, S-1, D_z) -> (B, S-1)
    z_l1_norm = torch.sum(torch.abs(latent_actions), dim=-1)
    
    # 4. Calculate the gated loss
    # Shape: (B, S-1)
    gated_loss = stillness_gate * z_l1_norm
    
    # 5. Return the final mean loss, scaled by lambda
    # We take the mean over the batch and sequence dimensions
    return torch.mean(gated_loss)


# def VIPloss(x):
#         """
#         Calculate the VIP loss for the HPC model.
#         """
#         batch_size = x.shape[0]
#         seq_len = x.shape[1]
#         # Sample (o_0, o_k, o_k+1, o_T) for VIP training
        
#         # Generate unique indices for each element in the batch
#         start_ind = torch.randint(0, seq_len-2, (batch_size, 1))
#         end_ind = torch.stack([torch.randint(start_ind[i].item()+1, seq_len, (1,)) for i in range(batch_size)]).squeeze(1)
#         s0_ind_vip = torch.stack([torch.randint(start_ind[i].item(), end_ind[i].item(), (1,)) for i in range(batch_size)]).squeeze(1)
#         s1_ind_vip = torch.clamp(s0_ind_vip + 1, max=end_ind)
        
#         # Extract different timesteps for each batch element using advanced indexing
#         batch_indices = torch.arange(batch_size)
#         e0 = x[batch_indices, start_ind.squeeze()]     # o_0 for each batch element
#         eg = x[batch_indices, end_ind]                 # o_g for each batch element
#         es0_vip = x[batch_indices, s0_ind_vip]         # o_t for each batch element
#         es1_vip = x[batch_indices, s1_ind_vip]         # o_t+1 for each batch element
        
#         # Self-supervised reward (this is always -1)
#         reward = (s0_ind_vip == end_ind).float() - 1

#         ## VIP Loss 
#         V_0 = sim(e0, eg)    # -||phi(s) - phi(g)||_2
#         r = reward.to(V_0.device)          # R(s;g) = (s==g) - 1 
#         V_s = sim(es0_vip, eg)
#         V_s_next = sim(es1_vip, eg)
#         V_loss = (1-self.gamma) * -V_0.mean() + torch.log(self.epsilon + torch.mean(torch.exp(-(r + self.gamma * V_s_next - V_s))))

#         # Optionally, add additional "negative" observations
#         V_s_neg = []
#         V_s_next_neg = []
#         for _ in range(self.num_negatives):
#             perm = torch.randperm(es0_vip.size()[0])
#             es0_vip_shuf = es0_vip[perm]
#             es1_vip_shuf = es1_vip[perm]

#             V_s_neg.append(sim(es0_vip_shuf, eg))
#             V_s_next_neg.append(sim(es1_vip_shuf, eg))

#         if self.num_negatives > 0:
#             V_s_neg = torch.cat(V_s_neg)
#             V_s_next_neg = torch.cat(V_s_next_neg)
#             r_neg = -torch.ones(V_s_neg.shape).to(V_0.device)
#             V_loss = V_loss + torch.log(self.epsilon + torch.mean(torch.exp(-(r_neg + self.gamma * V_s_next_neg - V_s_neg))))

#         return V_loss
