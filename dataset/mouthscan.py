import os
import json
import glob
import numpy as np
import torch
import open3d as o3d
import torchvision.transforms as T
import collections.abc as container_abcs

from PIL import Image

from utils.constants import IMG_MEAN, IMG_STD


TO_TENSOR_KEYS = ['input_coords_list', 'input_feats_list', 'action', 'action_normalized']


class MouthScanDataset(torch.utils.data.Dataset):
    """
    Dataset for next-step oral scan prediction from either full point clouds or mouth RGB images.
    """
    def __init__(
        self,
        data_root,
        episodes,
        input_type,
        target_type,
        num_action = 1,
        voxel_size = 0.005,
        image_size = 224,
        max_points = 60000,
        pos_pad = 0.01,
        joint_pad = 0.05
    ):
        assert input_type in ["full_pcd", "mouth_rgb"]
        assert target_type in ["tcp", "joint"]
        assert num_action >= 1

        self.data_root = data_root
        self.episodes = episodes
        self.input_type = input_type
        self.target_type = target_type
        self.num_action = num_action
        self.voxel_size = voxel_size
        self.image_size = image_size
        self.max_points = max_points
        self.pos_pad = pos_pad
        self.joint_pad = joint_pad

        self._json_cache = {}
        self.episode_files = {}
        self.samples = []

        for episode in episodes:
            episode_path = os.path.join(data_root, episode)
            files = sorted(glob.glob(os.path.join(episode_path, "data_*.json")))
            if not files:
                raise FileNotFoundError(f"No json files found in {episode_path}")
            self.episode_files[episode] = files
            for cur_idx in range(len(files) - 1):
                action_indices = [
                    min(cur_idx + 1 + k, len(files) - 1)
                    for k in range(num_action)
                ]
                self.samples.append((episode, cur_idx, action_indices))

        self.tcp_pos_min, self.tcp_pos_max, self.joint_min, self.joint_max = self._compute_stats()

        if self.input_type == "mouth_rgb":
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean = IMG_MEAN.tolist(), std = IMG_STD.tolist())
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.samples)

    def _load_json(self, path):
        cached = self._json_cache.get(path)
        if cached is not None:
            return cached
        with open(path, "r") as f:
            data = json.load(f)
        self._json_cache[path] = data
        return data

    def _compute_stats(self):
        tcp_positions = []
        joint_positions = []
        for files in self.episode_files.values():
            for path in files:
                record = self._load_json(path)
                tcp_positions.append(record["tcp_position_base"])
                joint_positions.append(record["joint_positions"])

        tcp_positions = np.array(tcp_positions, dtype = np.float32)
        joint_positions = np.array(joint_positions, dtype = np.float32)

        tcp_min = tcp_positions.min(axis = 0) - self.pos_pad
        tcp_max = tcp_positions.max(axis = 0) + self.pos_pad
        joint_min = joint_positions.min(axis = 0) - self.joint_pad
        joint_max = joint_positions.max(axis = 0) + self.joint_pad

        return tcp_min, tcp_max, joint_min, joint_max

    def _normalize_range(self, values, min_val, max_val):
        span = np.maximum(max_val - min_val, 1e-6)
        return (values - min_val) / span * 2.0 - 1.0

    def _load_full_pcd(self, episode_path, base_name):
        pcd_path = os.path.join(episode_path, "pcd", f"full_pcd_{base_name}.ply")
        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points, dtype = np.float32)
        colors = np.asarray(pcd.colors, dtype = np.float32)
        if colors.shape[0] == 0:
            colors = np.zeros_like(points)
        if self.max_points is not None and points.shape[0] > self.max_points:
            idx = np.random.choice(points.shape[0], self.max_points, replace = False)
            points = points[idx]
            colors = colors[idx]
        colors = (colors - IMG_MEAN) / IMG_STD
        feats = np.concatenate([points, colors], axis = -1)
        coords = np.ascontiguousarray(points / self.voxel_size, dtype = np.int32)
        feats = np.ascontiguousarray(feats, dtype = np.float32)
        return coords, feats

    def _load_mouth_rgb(self, episode_path, record):
        rel_path = record["mouth_camera"]["rgb_image_path"]
        rgb_path = os.path.join(episode_path, rel_path)
        image = Image.open(rgb_path).convert("RGB")
        return self.transform(image)

    def _load_actions(self, episode, action_indices):
        files = self.episode_files[episode]
        actions = []
        actions_normalized = []
        for idx in action_indices:
            record = self._load_json(files[idx])
            if self.target_type == "tcp":
                pos = np.array(record["tcp_position_base"], dtype = np.float32)
                quat = np.array(record["tcp_orientation_base_wxyz"], dtype = np.float32)
                action = np.concatenate([pos, quat], axis = -1)
                pos_norm = self._normalize_range(pos, self.tcp_pos_min, self.tcp_pos_max)
                action_norm = np.concatenate([pos_norm, quat], axis = -1)
            else:
                joints = np.array(record["joint_positions"], dtype = np.float32)
                action = joints
                action_norm = self._normalize_range(joints, self.joint_min, self.joint_max)
            actions.append(action)
            actions_normalized.append(action_norm)
        actions = np.stack(actions, axis = 0)
        actions_normalized = np.stack(actions_normalized, axis = 0)
        return actions, actions_normalized

    def __getitem__(self, index):
        episode, obs_idx, action_indices = self.samples[index]
        episode_path = os.path.join(self.data_root, episode)
        obs_path = self.episode_files[episode][obs_idx]
        obs_record = self._load_json(obs_path)
        base_name = os.path.splitext(os.path.basename(obs_path))[0]

        actions, actions_normalized = self._load_actions(episode, action_indices)

        ret_dict = {
            "action": torch.from_numpy(actions).float(),
            "action_normalized": torch.from_numpy(actions_normalized).float()
        }

        if self.input_type == "full_pcd":
            coords, feats = self._load_full_pcd(episode_path, base_name)
            ret_dict["input_coords_list"] = [coords]
            ret_dict["input_feats_list"] = [feats]
        else:
            ret_dict["input_rgb"] = self._load_mouth_rgb(episode_path, obs_record)

        return ret_dict


def collate_pcd(batch):
    import MinkowskiEngine as ME

    if isinstance(batch[0], container_abcs.Mapping):
        ret_dict = {}
        for key in batch[0]:
            if key in ["input_coords_list", "input_feats_list"]:
                ret_dict[key] = [d[key] for d in batch]
            elif key in TO_TENSOR_KEYS:
                ret_dict[key] = torch.stack([d[key] for d in batch], 0)
            else:
                ret_dict[key] = [d[key] for d in batch]
        coords_batch = ret_dict["input_coords_list"]
        feats_batch = ret_dict["input_feats_list"]
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_batch, feats_batch)
        ret_dict["input_coords_list"] = coords_batch
        ret_dict["input_feats_list"] = feats_batch
        return ret_dict

    raise TypeError("batch must contain dicts; found {}".format(type(batch[0])))
