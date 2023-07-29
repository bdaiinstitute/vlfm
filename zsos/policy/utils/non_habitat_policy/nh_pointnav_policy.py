from typing import Dict, Optional, Tuple

import torch
import torch.functional as F
import torch.nn as nn
from torch import Size

from .resnet import resnet18
from .rnn_state_encoder import LSTMStateEncoder


class ResNetEncoder(nn.Module):
    visual_keys = ["depth"]

    def __init__(self):
        super().__init__()
        self.running_mean_and_var = nn.Sequential()
        self.backbone = resnet18(1, 32, 32)
        self.compression = nn.Sequential(
            nn.Conv2d(
                256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.GroupNorm(1, 128, eps=1e-05, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(
        self, observations: Dict[str, torch.Tensor]
    ) -> torch.Tensor:  # type: ignore
        cnn_input = []
        for k in self.visual_keys:
            obs_k = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            obs_k = obs_k.permute(0, 3, 1, 2)
            cnn_input.append(obs_k)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class PointNavResNetNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.prev_action_embedding = nn.Linear(
            in_features=2, out_features=32, bias=True
        )
        self.tgt_embeding = nn.Linear(in_features=3, out_features=32, bias=True)
        self.visual_encoder = ResNetEncoder()
        self.visual_fc = nn.Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=2048, out_features=512, bias=True),
            nn.ReLU(inplace=True),
        )
        self.state_encoder = LSTMStateEncoder(576, 512, 2)
        self.num_recurrent_layers = self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        rnn_build_seq_info: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        x = []
        visual_feats = self.visual_encoder(observations)
        visual_feats = self.visual_fc(visual_feats)
        x.append(visual_feats)

        goal_observations = observations["pointgoal_with_gps_compass"]
        goal_observations = torch.stack(
            [
                goal_observations[:, 0],
                torch.cos(-goal_observations[:, 1]),
                torch.sin(-goal_observations[:, 1]),
            ],
            -1,
        )

        x.append(self.tgt_embeding(goal_observations))

        prev_actions = self.prev_action_embedding(masks * prev_actions.float())

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks, rnn_build_seq_info
        )

        return out, rnn_hidden_states, {}


class CustomNormal(torch.distributions.normal.Normal):
    def sample(self, sample_shape: Size = torch.Size()) -> torch.Tensor:
        return self.rsample(sample_shape)


class GaussianNet(nn.Module):
    min_log_std: int = -5
    max_log_std: int = 2
    log_std_init: float = 0.0

    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()
        num_linear_outputs = 2 * num_outputs

        self.mu_maybe_std = nn.Linear(num_inputs, num_linear_outputs)
        nn.init.orthogonal_(self.mu_maybe_std.weight, gain=0.01)
        nn.init.constant_(self.mu_maybe_std.bias, 0)
        nn.init.constant_(self.mu_maybe_std.bias[num_outputs:], self.log_std_init)

    def forward(self, x: torch.Tensor) -> CustomNormal:
        mu_maybe_std = self.mu_maybe_std(x).float()
        mu, std = torch.chunk(mu_maybe_std, 2, -1)

        mu = torch.tanh(mu)

        std = torch.clamp(std, self.min_std, self.max_std)
        std = torch.exp(std)

        return CustomNormal(mu, std, validate_args=False)


class PointNavResNetPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = PointNavResNetNet()
        self.action_distribution = GaussianNet(512, 2)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        features, rnn_hidden_states, _ = self.net(
            observations, rnn_hidden_states, prev_actions, masks
        )
        distribution = self.action_distribution(features)

        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()

        return action, rnn_hidden_states


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("state_dict_path", type=str, help="Path to state_dict file")
    args = parser.parse_args()

    ckpt = torch.load(args.state_dict_path, map_location="cpu")
    model = PointNavResNetPolicy()
    print(model)
    current_state_dict = model.state_dict()
    model.load_state_dict({k: v for k, v in ckpt.items() if k in current_state_dict})
    print("Loaded model from checkpoint successfully!")
