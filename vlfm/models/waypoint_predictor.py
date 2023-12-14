# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WaypointsPredictor(nn.Module):
    def __init__(
        self,
        map_input_type: str,
        text_encoder_type: str,
        use_obstacle_map: bool,
        process_feature_map: bool,
        map_encoder: nn.Module,
        text_encoder: nn.Module,
        attention_model: nn.Module,
        goal_pred_model: nn.Module,
        d_model: int,
        use_first_waypoint: bool,
        loss_norm: bool,
        text_init_dim: int = 768,
        device: Optional[Any] = None,
    ):
        super(WaypointsPredictor, self).__init__()

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.text_encoder_type = text_encoder_type

        self.map_encoder = map_encoder
        self.text_encoder = text_encoder
        self.text_postprocess = nn.Linear(
            in_features=text_init_dim, out_features=d_model
        ).to(
            self.device
        )  # to produce same size embeddings as the map
        self.attention_model = attention_model
        self.goal_pred_model = goal_pred_model

        if loss_norm:
            self.position_mse_loss = nn.MSELoss()
        else:
            self.position_mse_loss = nn.MSELoss(reduction="sum")

        self.d_model = d_model
        self.use_first_waypoint = use_first_waypoint

        if self.use_first_waypoint:
            self.conv_point1 = nn.Conv2d(d_model + 1, d_model, 1, 1, padding=0).to(
                self.device
            )

    def forward(
        self, batch: Dict[str, torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor]:
        vl_map = batch["map_vl"]

        # bert inputs
        if self.text_encoder_type == "BERT":
            tokens_tensor = batch["tokens_tensor"].long()
            segments_tensors = batch["segments_tensors"].long()
            tokens_tensor = tokens_tensor.squeeze(1)
            segments_tensors = segments_tensors.squeeze(1)

            outputs = self.bert_model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

            text_feat_list = []
            for b in range(tokens_tensor.shape[0]):
                text_feat_list.append(hidden_states[-1][b][:])
            text_feat = torch.stack(text_feat_list)
        else:
            text_feat = self.text_encoder.encode_text(batch["tokens_tensor"]).float()
            text_feat.to(self.device)
            text_feat

        B, T, C, H, W = vl_map.shape  # batch, sequence, 1, height, width
        vl_map = vl_map.view(B * T, C, H, W)
        vl_map.requires_grad = True
        text_feat.requires_grad = True

        ### Prepare the input embeddings of the map and text
        encoded_map = self.map_encoder(vl_map)  # B x 128 x 32 x 32
        map_enc_res = encoded_map.shape[2]

        encoded_map_in = encoded_map.permute(0, 2, 3, 1).view(
            encoded_map.shape[0], -1, self.d_model
        )
        encoded_text = self.text_postprocess(text_feat)

        # replicate encoded text for number of encoded_map_in
        if T > 1:
            encoded_text = encoded_text.unsqueeze(1).repeat(1, T, 1, 1)
            encoded_text = encoded_text.view(
                B * T, encoded_text.shape[2], encoded_text.shape[3]
            )

        ### Apply attention between the map and text
        dec_out, dec_enc_attns = self.attention_model(
            enc_inputs=encoded_text, dec_inputs=encoded_map_in
        )  # B x 1024 x 128

        ### Use the attention-augmented map embeddings to predict the waypoints
        dec_out = dec_out.permute(0, 2, 1).view(
            dec_out.shape[0], -1, map_enc_res, map_enc_res
        )  # reshape map attn # B x 128 x 32 x 32

        if self.use_first_waypoint:
            point_heatmap = batch["goal_heatmap"][:, :, 0, :, :]  # B x T x 1 x 48 x 48
            point_heatmap = F.interpolate(
                point_heatmap, size=(dec_out.shape[2], dec_out.shape[3]), mode="nearest"
            )
            point_heatmap = point_heatmap.view(
                B * T, point_heatmap.shape[2], point_heatmap.shape[3]
            ).unsqueeze(1)
            pred_in = torch.cat((dec_out, point_heatmap), dim=1)
            pred_in = self.conv_point1(pred_in)
        else:
            pred_in = dec_out

        waypoints_heatmaps, waypoints_cov_raw = self.goal_pred_model(pred_in)

        # get the prob distribution over uncovered/covered for each waypoint
        waypoints_cov = F.softmax(waypoints_cov_raw, dim=2)

        return waypoints_heatmaps, waypoints_cov
