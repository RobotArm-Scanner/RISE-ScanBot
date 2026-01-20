import os
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist

from copy import deepcopy
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy import RISE, RGBPolicy
from dataset.mouthscan import MouthScanDataset, collate_pcd
from utils.training import set_seed, plot_history, sync_loss


INPUT_TYPE = "mouth_rgb"
TARGET_TYPE = "tcp"


default_args = edict({
    "data_root": "/scratch/refracta1342/datasets",
    "episodes": [
        "episode_e2_f1__siwon2_augmented01",
        "episode_e2_f1__siwon2_augmented02",
        "episode_e2_f1__siwon2_augmented03",
        "episode_e2_f1__siwon2_augmented04",
        "episode_e2_f1__siwon2_augmented05",
        "episode_e2_f1__siwon2_augmented06",
        "episode_e2_f1__siwon2_augmented07",
        "episode_e2_f1__siwon2_augmented08",
        "episode_e2_f1__siwon2_augmented09",
        "episode_e2_f1__siwon2_augmented10"
    ],
    "num_action": 1,
    "voxel_size": 0.005,
    "image_size": 224,
    "max_points": 60000,
    "pos_pad": 0.01,
    "joint_pad": 0.05,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "ckpt_dir": "logs/mouthscan_rgb_tcp",
    "resume_ckpt": None,
    "resume_epoch": -1,
    "lr": 3e-4,
    "batch_size": 64,
    "num_epochs": 300,
    "save_epochs": 25,
    "num_workers": 8,
    "seed": 233
})


def train(args_override):
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    args.episodes = [e.strip() for e in args.episodes.split(",") if e.strip()]

    action_dim = 7 if TARGET_TYPE == "tcp" else 6

    torch.multiprocessing.set_sharing_strategy("file_system")
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    os.environ["NCCL_P2P_DISABLE"] = "1"
    dist.init_process_group(backend = "nccl", init_method = "env://", world_size = world_size, rank = rank)

    set_seed(args.seed)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if rank == 0:
        print("Loading dataset ...")

    dataset = MouthScanDataset(
        data_root = args.data_root,
        episodes = args.episodes,
        input_type = INPUT_TYPE,
        target_type = TARGET_TYPE,
        num_action = args.num_action,
        voxel_size = args.voxel_size,
        image_size = args.image_size,
        max_points = args.max_points,
        pos_pad = args.pos_pad,
        joint_pad = args.joint_pad
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas = world_size,
        rank = rank,
        shuffle = True
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size // world_size,
        num_workers = args.num_workers,
        collate_fn = collate_pcd if INPUT_TYPE == "full_pcd" else None,
        sampler = sampler,
        drop_last = True
    )

    if rank == 0:
        print("Loading policy ...")

    if INPUT_TYPE == "full_pcd":
        policy = RISE(
            num_action = args.num_action,
            input_dim = 6,
            obs_feature_dim = args.obs_feature_dim,
            action_dim = action_dim,
            hidden_dim = args.hidden_dim,
            nheads = args.nheads,
            num_encoder_layers = args.num_encoder_layers,
            num_decoder_layers = args.num_decoder_layers,
            dropout = args.dropout
        ).to(device)
    else:
        policy = RGBPolicy(
            num_action = args.num_action,
            action_dim = action_dim,
            obs_feature_dim = args.obs_feature_dim,
            backbone = "resnet18"
        ).to(device)

    if rank == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))

    policy = nn.parallel.DistributedDataParallel(
        policy,
        device_ids = [local_rank],
        output_device = local_rank,
        find_unused_parameters = True
    )

    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        if rank == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    if rank == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    if rank == 0:
        print("Loading optimizer and scheduler ...")

    optimizer = torch.optim.AdamW(policy.parameters(), lr = args.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 2000,
        num_training_steps = len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    train_history = []
    policy.train()

    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        if rank == 0:
            print("Epoch {}".format(epoch))
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        avg_loss = 0.0

        for data in dataloader:
            action_data = data["action_normalized"].to(device)
            if INPUT_TYPE == "full_pcd":
                import MinkowskiEngine as ME
                cloud_coords = data["input_coords_list"].to(device)
                cloud_feats = data["input_feats_list"].to(device)
                cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
                loss = policy(cloud_data, action_data, batch_size = action_data.shape[0])
            else:
                images = data["input_rgb"].to(device)
                loss = policy(images, action_data)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            avg_loss += loss.item()

        avg_loss = avg_loss / max(num_steps, 1)
        sync_loss(avg_loss, device)
        train_history.append(avg_loss)

        if rank == 0:
            print("Train loss: {:.6f}".format(avg_loss))
            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
                )
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    if rank == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type = str, default = default_args["data_root"])
    parser.add_argument("--episodes", type = str, required = False, default = ",".join(default_args["episodes"]))
    parser.add_argument("--num_action", type = int, default = default_args["num_action"])
    parser.add_argument("--voxel_size", type = float, default = default_args["voxel_size"])
    parser.add_argument("--image_size", type = int, default = default_args["image_size"])
    parser.add_argument("--max_points", type = int, default = default_args["max_points"])
    parser.add_argument("--pos_pad", type = float, default = default_args["pos_pad"])
    parser.add_argument("--joint_pad", type = float, default = default_args["joint_pad"])
    parser.add_argument("--obs_feature_dim", type = int, default = default_args["obs_feature_dim"])
    parser.add_argument("--hidden_dim", type = int, default = default_args["hidden_dim"])
    parser.add_argument("--nheads", type = int, default = default_args["nheads"])
    parser.add_argument("--num_encoder_layers", type = int, default = default_args["num_encoder_layers"])
    parser.add_argument("--num_decoder_layers", type = int, default = default_args["num_decoder_layers"])
    parser.add_argument("--dim_feedforward", type = int, default = default_args["dim_feedforward"])
    parser.add_argument("--dropout", type = float, default = default_args["dropout"])
    parser.add_argument("--ckpt_dir", type = str, required = True)
    parser.add_argument("--resume_ckpt", type = str, default = default_args["resume_ckpt"])
    parser.add_argument("--resume_epoch", type = int, default = default_args["resume_epoch"])
    parser.add_argument("--lr", type = float, default = default_args["lr"])
    parser.add_argument("--batch_size", type = int, default = default_args["batch_size"])
    parser.add_argument("--num_epochs", type = int, default = default_args["num_epochs"])
    parser.add_argument("--save_epochs", type = int, default = default_args["save_epochs"])
    parser.add_argument("--num_workers", type = int, default = default_args["num_workers"])
    parser.add_argument("--seed", type = int, default = default_args["seed"])

    args = parser.parse_args()
    train(vars(args))
