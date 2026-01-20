import torch
import torch.nn as nn
import torchvision.models as models

from policy.diffusion import DiffusionUNetPolicy


class RGBPolicy(nn.Module):
    """
    RGB encoder + diffusion action decoder for next-step prediction.
    """
    def __init__(
        self,
        num_action = 1,
        action_dim = 7,
        obs_feature_dim = 512,
        backbone = "resnet18"
    ):
        super().__init__()

        if backbone == "resnet18":
            encoder = models.resnet18(weights = None)
            encoder.fc = nn.Identity()
            encoder_dim = 512
        elif backbone == "resnet34":
            encoder = models.resnet34(weights = None)
            encoder.fc = nn.Identity()
            encoder_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.encoder = encoder
        if encoder_dim != obs_feature_dim:
            self.proj = nn.Linear(encoder_dim, obs_feature_dim)
        else:
            self.proj = nn.Identity()

        self.action_decoder = DiffusionUNetPolicy(
            action_dim = action_dim,
            horizon = num_action,
            n_obs_steps = 1,
            obs_feature_dim = obs_feature_dim
        )

    def forward(self, images, actions = None):
        features = self.encoder(images)
        readout = self.proj(features)
        if actions is not None:
            loss = self.action_decoder.compute_loss(readout, actions)
            return loss
        with torch.no_grad():
            action_pred = self.action_decoder.predict_action(readout)
        return action_pred
