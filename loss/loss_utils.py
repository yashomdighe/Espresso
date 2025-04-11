import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torch_geometric.nn import radius_graph

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def _query_ball_point(pointcloud, radius):
    """
    Given a pointcloud (N,3), returns:
      neighbor_indices: (N, M) long tensor, where M is the maximum number of neighbors found.
      mask: (N, M) bool tensor indicating valid neighbor indices.
    
    This simple implementation computes all pairwise distances.
    """
    N = pointcloud.size(0)
    # Compute pairwise Euclidean distances (N, N)
    dists = torch.cdist(pointcloud, pointcloud)  
    # Determine neighbors within the radius (including self)
    neighbor_mask = dists <= radius  # (N, N)
    
    # For each point, find neighbor indices and pad to the maximum number found
    max_neighbors = int(neighbor_mask.sum(dim=1).max().item())
    neighbor_indices_list = []
    valid_mask_list = []
    for i in range(N):
        indices = torch.nonzero(neighbor_mask[i]).squeeze(1)
        num_neighbors = indices.numel()
        # If there are fewer neighbors than max, pad the indices
        if num_neighbors < max_neighbors:
            pad = indices.new_full((max_neighbors - num_neighbors,), indices[0])
            indices = torch.cat([indices, pad], dim=0)
            valid = torch.cat([torch.ones(num_neighbors, dtype=torch.bool),
                               torch.zeros(max_neighbors - num_neighbors, dtype=torch.bool)])
        else:
            valid = torch.ones(max_neighbors, dtype=torch.bool)
        neighbor_indices_list.append(indices.unsqueeze(0))
        valid_mask_list.append(valid.unsqueeze(0))
    
    neighbor_indices = torch.cat(neighbor_indices_list, dim=0)  # (N, M)
    mask = torch.cat(valid_mask_list, dim=0)  # (N, M)
    return neighbor_indices, mask

def rigid_loss(input_means, deltas, radius=1e-2):
    """
    input_means: [B, N, 3]
    deltas: [B, N, 3]
    """
    B, N, _ = input_means.shape
    device = input_means.device

    # Flatten the batch for radius_graph
    flat_means = input_means.reshape(B * N, 3)
    flat_deltas = deltas.reshape(B * N, 3)

    # Batch vector: [0, 0, ..., 1, 1, ..., B-1, ..., B-1]
    batch_vec = torch.arange(B, device=device).repeat_interleave(N)

    # Compute radius graph (batch-aware)
    edge_index = radius_graph(flat_means, r=radius, batch=batch_vec, loop=False)

    # Get differences of deltas at edges
    src, dst = edge_index
    diff = flat_deltas[src] - flat_deltas[dst]

    # Compute squared differences
    loss_rigid = diff.pow(2).sum(dim=1).mean() + 1e-10
    return loss_rigid

def rigidity_loss2(neighbors_list, predictions):
    """
    Computes an RMSE loss that enforces the predictions for each point to be similar to its neighbors.
    The neighbor indices are precomputed as a list (from NumPy).
    
    Args:
      neighbors_list: list of length N with neighbor indices for each point.
      predictions: (N, 3) torch tensor of predicted offsets.
    
    Returns:
      rmse: a scalar torch tensor representing the RMSE loss.
    """
    total_sq_error = 0.0
    total_count = 0

    # Loop over each point and compute the squared error with all its neighbors.
    for i, nbr_indices in enumerate(neighbors_list):
        # Skip if no neighbors found (should not happen if self is included)
        if len(nbr_indices) == 0:
            continue
        # Compute differences between the prediction for point i and each neighbor's prediction.
        # predictions[nbr_indices] returns a tensor of shape (num_neighbors, 3)
        diff = predictions[i].unsqueeze(0) - predictions[nbr_indices]
        total_sq_error += diff.pow(2).sum()
        total_count += len(nbr_indices)
    
    mse_loss = total_sq_error / total_count if total_count > 0 else torch.tensor(0.0, device=predictions.device)
    # rmse_loss = torch.sqrt(mse_loss)
    return mse_loss

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)