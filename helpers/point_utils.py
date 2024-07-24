import torch
import numpy as np
from openpoints.models.layers import furthest_point_sample as farthest_point_sample, random_sample

def index_points(points, idx):
    # Adapted from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    C = points.shape[-1]
    if len(idx.shape) == 2:
        new_points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, C))
    elif len(idx.shape) == 3:
        B, K, S = idx.shape
        idx = idx.reshape(B, -1)
        new_points = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, C))
        new_points = new_points.reshape(B, K, S, C)
    return new_points

def determine_sampling_points(num_points, ratio=1200/1024):
    """Determine the total number of points to sample based on the requested number."""
    return int(num_points * ratio)

def filter_and_sample_points(image_features, pcd, feat_size, num_points, bs, bounds, rand_sample, resample, pcd_bound_masks):
    # Flatten image features from different views into a single tensor
    flat_image_features = torch.cat([
        p.permute(0, 2, 3, 1).reshape(bs, p.shape[2]*p.shape[3], feat_size) for p in image_features
    ], 1) # Shape: [bs, H * W * n_cameras, C]
    # Flatten point cloud data
    pcd_flat = torch.cat([
        p.permute(0, 2, 3, 1).reshape(bs, -1, 3) for p in pcd
    ], 1)
    
    if pcd_bound_masks is not None:
        pcd_bound_masks_flat = torch.cat([
            p.reshape(bs, -1) for p in pcd_bound_masks
        ], 1) # Shape: [bs, H * W * n_cameras]
    
    xyz = pcd_flat.permute(0, 2, 1) # (bs, 3, N)
    feature = flat_image_features.permute(0, 2, 1) # (bs, C, N)

    # Filter points outside of bounds
    all_indices = []
    for i in range(bs):
        if pcd_bound_masks is not None:
            indices = pcd_bound_masks_flat[i].nonzero().squeeze(-1)
        else:
            indices = (
                (xyz[i, 0, :] >= bounds[0, 0]) *
                (xyz[i, 1, :] >= bounds[0, 1]) *
                (xyz[i, 2, :] >= bounds[0, 2]) *
                (xyz[i, 0, :] <= bounds[0, 3]) *
                (xyz[i, 1, :] <= bounds[0, 4]) *
                (xyz[i, 2, :] <= bounds[0, 5])
            ).nonzero().squeeze(-1)

        all_indices.append(indices)
    
    indices_len = [len(indices) for indices in all_indices]
    for i in range(bs):
        num_pad = np.max(indices_len) - indices_len[i]
        if np.max(indices_len) == 0:
            # all batch of points are outside of bounds
            all_indices[i] = torch.arange(xyz.shape[2]).to(xyz.device)
        else:
            if indices_len[i] > 0:
                all_indices[i] = torch.cat([
                    all_indices[i], all_indices[i][np.random.randint(indices_len[i], size=num_pad)]
                    ], dim=0)
            else:
                # all of points are outside of bounds
                all_indices[i] = torch.randperm(xyz.shape[2])[:num_pad].to(xyz.device)

    # Gather points and features based on the indices
    xyz_list, feature_list = [], []
    for i in range(bs):
        xyz_list.append(xyz[i,:,:].index_select(dim=-1, index=all_indices[i]).unsqueeze(dim=0))
        feature_list.append(feature[i,:,:].index_select(dim=-1, index=all_indices[i]).unsqueeze(dim=0))
    xyz = torch.cat(xyz_list, dim=0)
    feature = torch.cat(feature_list, dim=0)

    xyz = xyz.permute(0, 2, 1) # (bs, N, 3)
    feature = feature.permute(0, 2, 1)

    # Downsample the points
    if num_points < xyz.shape[1]:
        if rand_sample:
            # ramdom sample
            index = random_sample(xyz, num_points)
        else:
            # farthest point sample
            xyz = xyz.contiguous()
            if resample:
                point_all = determine_sampling_points(num_points) # choose a slightly larger number of points to increase randomness
                if xyz.shape[1] < point_all:
                    point_all = xyz.shape[1]
                index = farthest_point_sample(xyz, point_all).long()
                index = index[:, np.random.choice(point_all, num_points, False)]
            else:
                index = farthest_point_sample(xyz, num_points).long()
        xyz = index_points(xyz, index)
        feature = index_points(feature, index)
    else:
        index = random_sample(xyz, num_points)
        xyz = index_points(xyz, index)
        feature = index_points(feature, index)
    
    return xyz, feature 