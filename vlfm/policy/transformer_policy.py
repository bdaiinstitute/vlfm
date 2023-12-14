# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Dict, List, Optional, Tuple

import clip
import numpy as np
import torch
import torch.nn as nn
from CM2.models.networks import MapAttention, MapEncoder, ResNetUNetGoalPred

from vlfm.models.waypoint_predictor import WaypointsPredictor
from vlfm.text_processing.utils import parse_instruction
from vlfm.utils.geometry_utils import get_rotation_matrix
from vlfm.vlm.lseg import LSeg

from .path_policy import BasePathPolicy


class TransformerPolicy(BasePathPolicy):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        eval = False
        loss_norm = False
        use_value_map = False  # True
        device = None

        torch.set_grad_enabled(True)

        self._vl_map.use_direction_embedding = False

        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

        self.is_eval = eval
        self.use_value_map = use_value_map

        self.ego_map_shape = 64
        self.use_first_waypoint = True

        self.weight_obstacle_loss = 0.2
        self.weight_path_loss = 5.0
        # self.weight_position_loss = 0.5
        self.weight_final_dist_loss = 10.0
        # self.weight_final_path_loss = 0.5

        self.weight_dist_loss = 1.0

        self.weight_confidence_loss = 1.0

        self.gt_path_tensor: torch.tensor = torch.tensor([0.0, 0.0], device=self.device)
        self.gt_path_tensor_px: torch.tensor = torch.tensor(
            [0.0, 0.0], device=self.device
        )

        self.mse = nn.MSELoss()
        self.smoothL1 = nn.SmoothL1Loss(beta=3.0)  # set beta to goal success distance

        if loss_norm:
            self.position_mse_loss = nn.MSELoss()
        else:
            self.position_mse_loss = nn.MSELoss(reduction="sum")

        self.prev_conf = 0.0
        self.conf_thresh = 0.3

        # Make model
        d_model = 128
        num_waypoints = 5
        if self.use_first_waypoint:
            out_channels = num_waypoints - 1
        else:
            out_channels = num_waypoints

        self.models_dict: Dict[str, nn.Module] = {}

        # We should use re-use CLIP from our map to avoid loading it twice
        # Note we are keeping all of CLIP frozen
        assert isinstance(self._vl_map._vl_model, LSeg)
        text_encoder = self._vl_map._vl_model.clip_model  # Lseg only
        if use_value_map:
            map_encoder = MapEncoder(n_channel_in=1, n_channel_out=d_model).to(
                self.device
            )
        else:
            map_encoder = MapEncoder(n_channel_in=512, n_channel_out=d_model).to(
                self.device
            )
        attention_model = MapAttention(
            d_model=d_model,
            d_ff=64,
            d_k=16,
            d_v=16,
            n_heads=8,
            n_layers=4,
            device=device,
        ).to(self.device)

        goal_pred_model = ResNetUNetGoalPred(
            n_channel_in=d_model, n_class_out=out_channels, with_lstm=False
        ).to(self.device)

        if use_value_map:
            self.map_input_type = "value"
        else:
            self.map_input_type = "feature"

        self.model = WaypointsPredictor(
            map_input_type=self.map_input_type,
            text_encoder_type="CLIP",
            use_obstacle_map=False,
            process_feature_map=False,
            map_encoder=map_encoder,
            text_encoder=text_encoder,
            attention_model=attention_model,
            goal_pred_model=goal_pred_model,
            d_model=d_model,
            use_first_waypoint=True,
            loss_norm=False,
            text_init_dim=512,
        )

        if not self.is_eval:
            for param in self.model.map_encoder.parameters():
                param.requires_grad = True
            for param in self.model.attention_model.parameters():
                param.requires_grad = True
            for param in self.model.goal_pred_model.parameters():
                param.requires_grad = True
            for param in self.model.text_encoder.parameters():
                param.requires_grad = False
            for param in self.model.text_postprocess.parameters():
                param.requires_grad = True
            if self.use_first_waypoint:
                for param in self.model.conv_point1.parameters():
                    param.requires_grad = True

        self.models_dict["goal_pred_model"] = self.model  # goal_pred_model
        # self.models_dict['map_encoder'] = map_encoder

        self.optimizers_dict: Dict[str, torch.optim.Optimizer] = {}
        for model in self.models_dict:
            self.optimizers_dict[model] = torch.optim.Adam(
                [
                    {
                        "params": self.models_dict[model].parameters(),
                        "initial_lr": self.args.transformer.lr,
                    }
                ],
                lr=self.args.transformer.lr,
                betas=(self.args.transformer.beta1, 0.999),
            )

        # First goal should just be [0,0] as it's ego-centric...?
        self.goal_heatmap = (
            self.path_to_heatmap(torch.zeros(1, 2, device=self.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )

    def _reset(self) -> None:
        super()._reset()

    def set_gt_path_for_viz(
        self, gt_path_for_viz: np.ndarray, gt_path_world_coord: np.ndarray
    ) -> None:
        self.gt_path_for_viz = gt_path_for_viz
        self.gt_path_world_coord = gt_path_world_coord

        # Convert to px
        gt_path_px = self._vl_map._xy_to_px(gt_path_for_viz)

        # Save as tensor for losses
        self.gt_path_tensor = torch.tensor(
            gt_path_for_viz, device=self.device, dtype=torch.float32
        )
        self.gt_path_tensor_px = torch.tensor(
            gt_path_px, device=self.device, dtype=torch.float32
        )

    def _parse_instruction(self, instruction: str) -> List[str]:
        parsed_instruct = parse_instruction(
            instruction, split_strs=["\r\n", "\n", ".", ",", " and ", " then "]
        )

        print("PARSING: ", instruction)
        print("OUPUT: ", parsed_instruct)
        return parsed_instruct

    def _plan_loop(self) -> Tuple[np.ndarray, bool, bool]:
        # print("GT GOAL: ", self.gt_path_for_viz[-1, :])

        robot_xy = self._observations_cache["robot_xy"]
        yaw = self._observations_cache["robot_heading"]

        stop = False

        if len(self._instruction_parts) > (self._curr_instruction_idx + 1):
            last_instruction = False
        else:
            last_instruction = True

        stop = False
        switch = False

        # Call model to plan
        ego_map = self._vl_map.get_egomap(robot_xy, yaw, self.ego_map_shape)
        if self.use_value_map:
            text = self._instruction_parts[self._curr_instruction_idx]
            text_embed = self._vl_map._vl_model.get_text_embedding(text, head="embed")

            ego_map1 = torch.tensor(
                self._vl_map._vl_model.get_similarity_batch(
                    ego_map.reshape(
                        [ego_map.shape[0] * ego_map.shape[1]]
                        + list(self._vl_map._feats_sz)
                    ),
                    text_embed,
                ).reshape([1, ego_map.shape[0], ego_map.shape[1]]),
                device=self.device,
            )
        else:
            ego_map1 = ego_map.permute(2, 0, 1)

        # Get predicted path for current instruction part
        batch: Dict[str, torch.tensor] = {}
        batch["map_vl"] = ego_map1.unsqueeze(0).unsqueeze(0)
        batch["goal_heatmap"] = self.goal_heatmap
        batch["instruction_part"] = self._instruction_parts[self._curr_instruction_idx]
        batch["tokens_tensor"] = clip.tokenize(
            self._instruction_parts[self._curr_instruction_idx],
            context_length=77,
            truncate=True,
        ).to(self.device)
        predicted_wps_curr, _ = self.model(batch)
        conf_curr = torch.max(predicted_wps_curr[:])

        if last_instruction:
            # Decide whether to stop
            stop = conf_curr < self.prev_conf
            self.prev_conf = conf_curr

        else:
            if self.use_value_map:
                text = self._instruction_parts[self._curr_instruction_idx + 1]
                text_embed = self._vl_map._vl_model.get_text_embedding(
                    text, head="embed"
                )
                ego_map2 = torch.tensor(
                    self._vl_map._vl_model.get_similarity_batch(
                        ego_map.reshape(
                            [ego_map.shape[0] * ego_map.shape[1]]
                            + list(self._vl_map._feats_sz)
                        ),
                        text_embed,
                    ).reshape([1, ego_map.shape[0], ego_map.shape[1]]),
                    device=self.device,
                )
            else:
                ego_map2 = ego_map1

            batch2: Dict[str, torch.tensor] = {}
            batch2["map_vl"] = ego_map2.unsqueeze(0).unsqueeze(0)
            batch2["goal_heatmap"] = self.goal_heatmap
            batch2["instruction_part"] = self._instruction_parts[
                self._curr_instruction_idx + 1
            ]
            batch2["tokens_tensor"] = clip.tokenize(
                self._instruction_parts[self._curr_instruction_idx + 1],
                context_length=77,
                truncate=True,
            ).to(self.device)
            # Get predicted path for next instruction part
            predicted_wps_next, _ = self.model(batch2)
            conf_next = torch.max(predicted_wps_next[:])

            # Decide whether to switch
            switch = (conf_next > conf_curr) or (conf_curr < self.prev_conf)

            # print(
            #     "SWITCH DECISION: ", switch.item(), conf_curr.item(), conf_next.item()
            # )

            if switch:
                self.prev_conf = conf_next
                self._curr_instruction_idx += 1
            else:
                self.prev_conf = conf_curr

        if switch:
            predicted_wps = predicted_wps_next.squeeze()
        else:
            predicted_wps = predicted_wps_curr.squeeze()

        # convert predicted waypoints to global frame xy and save
        path_init, _ = self.heatmap_to_path(predicted_wps)

        # Remove zeros
        path = path_init[torch.abs(path_init).sum(dim=1) != 0, :]

        for model in self.models_dict:
            self.optimizers_dict[model].zero_grad()

        if path.shape[0] == 0:
            # Confidence loss to force it not to predict low all the time...
            loss = self.get_loss_confidence(predicted_wps)
            # need to save graph because we have a loss at the end which uses it
            loss.backward()
            print("LOSS confidence: ", loss)
            self.optimizers_dict["goal_pred_model"].step()

            return self._plan_loop()

        if not self.is_eval:
            # Apply loss

            # if last_instruction:
            final_pred = predicted_wps[torch.abs(path_init).sum(dim=1) != 0, ...][
                -1, ...
            ].view(1, 16, 16)

            gt_path = self.gt_path_tensor_px.clone()[-1, :].unsqueeze(0)
            pos = torch.tensor(
                self._vl_map._xy_to_px(robot_xy.reshape(1, 2))[:], device=self.device
            ).reshape(2)
            R = torch.tensor(
                get_rotation_matrix(-yaw), device=self.device, dtype=torch.float32
            )
            gt_path[:, 0] -= pos[0]
            gt_path[:, 1] -= pos[1]
            gt_path = (R @ gt_path.T).T

            final_gt = self.path_to_heatmap(gt_path).detach()

            loss = self.get_loss_final(
                predicted_wps, final_pred, final_gt, robot_xy, yaw, last_instruction
            )
            print("LOSS on final: ", loss)
            loss.backward()
            self.optimizers_dict["goal_pred_model"].step()

            # else:
            #     loss = self.get_loss(predicted_wps, robot_xy, yaw)
            #     loss.backward()
            #     print("LOSS: ", loss)
            #     self.optimizers_dict['goal_pred_model'].step()

        # Save
        for model in self.models_dict:
            torch.save(self.models_dict[model].state_dict(), f"data/{model}.pth")

        if stop:
            return robot_xy, True, False

        path = path.clone().detach().cpu().numpy()

        R = get_rotation_matrix(yaw)
        path = (R @ path.T).T

        robot_xy_px = self._vl_map._xy_to_px(robot_xy.reshape(1, 2)).reshape(2)

        path[:, 0] = path[:, 0] + robot_xy_px[0]
        path[:, 1] = path[:, 1] + robot_xy_px[1]

        path = self._vl_map._px_to_xy(path)

        self._path_to_follow = path

        return self._path_to_follow[0], stop, False

    def _plan(
        self,
    ) -> Tuple[np.ndarray, bool, bool]:  # next waypoint, stop, pause and turn
        if self._vl_map.enable_stairs:
            self._stair_preplan_step()

        replan, force_dont_stop, idx_path = self._pre_plan_logic()

        ###Path planning
        if replan:
            return self._plan_loop()
        return self._path_to_follow[idx_path], False, False

    def get_corresponding_waypoints(
        self,
        pred_waypoints: torch.tensor,
        ego_centric: bool = False,
        agent_pos: Optional[np.ndarray] = None,
        agent_yaw: Optional[float] = None,
    ) -> torch.tensor:
        """Get GT waypoints closest to predicted waypoints"""
        gt_path = self.gt_path_tensor_px.clone()

        # Convert GT path to ego-centric
        if ego_centric:
            assert not (agent_pos is None or agent_yaw is None)
            pos = torch.tensor(
                self._vl_map._xy_to_px(agent_pos.reshape(1, 2))[:], device=self.device
            ).reshape(2)
            R = torch.tensor(
                get_rotation_matrix(-agent_yaw), device=self.device, dtype=torch.float32
            )
            gt_path[:, 0] -= pos[0]
            gt_path[:, 1] -= pos[1]
            gt_path = (R @ gt_path.T).T

        # Want to penalize back-tracking,
        # so only allow to get GT waypoints same or after previously chosen
        corresp = torch.zeros((pred_waypoints.shape[0], 2), device=self.device)
        idx_prev = 0
        for i in range(pred_waypoints.shape[0]):
            dists = torch.sqrt(
                torch.sum(
                    torch.square(
                        pred_waypoints[i, :]
                        .reshape(1, 2)
                        .repeat(gt_path.shape[0] - idx_prev - 1, 1)
                        - gt_path[idx_prev + 1, :]
                    ),
                    dim=1,
                )
            )
            idx = torch.argmin(dists)
            corresp[i, :] = gt_path[idx_prev + idx + 1, :]
            idx_prev += idx
        return corresp

    # Adapted from https://github.com/ggeorgak11/CM2/blob/master/datasets/util/utils.py
    def path_to_heatmap(
        self, path: torch.tensor, out_size: List[int] = [16, 16], sigma: float = 1.0
    ) -> torch.tensor:
        # Note that out_size should match hourglass output size

        x_scale = out_size[0] / self.ego_map_shape
        y_scale = out_size[1] / self.ego_map_shape
        x = torch.arange(0, out_size[1], dtype=torch.float32, device=self.device)
        y = torch.arange(0, out_size[0], dtype=torch.float32, device=self.device)
        yg, xg = torch.meshgrid(y, x)

        gaussian_hm = torch.zeros(
            (path.shape[0], out_size[0], out_size[1]), device=self.device
        )
        for i, keypoint in enumerate(path):
            kp_x = keypoint[0] * x_scale
            kp_y = keypoint[1] * y_scale
            gaussian_hm[i, :, :] = torch.exp(
                -((xg - kp_x) ** 2 + (yg - kp_y) ** 2) / (2 * sigma**2)
            )
        return gaussian_hm

    # Adapted from https://github.com/ggeorgak11/CM2/blob/master/datasets/util/utils.py
    def heatmap_to_path(self, heatmaps: torch.tensor) -> torch.tensor:
        vals, uv = torch.max(
            heatmaps.view(heatmaps.shape[0], heatmaps.shape[1] * heatmaps.shape[2]), 1
        )
        # zero out entries below the detection threshold
        uv *= (vals > self.conf_thresh).type(torch.long)
        vals *= (vals > self.conf_thresh).type(torch.long)
        rows = uv / heatmaps.shape[2]
        cols = uv % heatmaps.shape[2]
        return torch.stack([cols, rows], 1).type(torch.float), vals

    # Adapted from https://github.com/ggeorgak11/CM2/blob/master/models/vln_goal_predictor.py
    def path_loss(
        self,
        pred_heatmap: torch.tensor,
        gt_heatmap: torch.tensor,  # num_waypoints x h x w
    ) -> torch.tensor:
        # TODO: should we do some masking based on if things are visible like in CM2?

        loss = self.position_mse_loss(pred_heatmap, gt_heatmap)
        return loss

    # def position_loss(self, final_wp: torch.tensor, gt_wp: torch.tensor) -> float:
    #     return self.mse(final_wp.reshape(-1,2), gt_wp.reshape(-1,2))

    def obstacle_loss(
        self, pred_waypoints: torch.tensor, obstacle_map: torch.tensor
    ) -> torch.tensor:
        """Penalize predicting waypoint on obstacle"""
        # probably only makes sense when obstacle map is an input to the model...
        pass

    # def final_dist_loss(self, final_loc: torch.tensor, goal: torch.tensor) -> float:
    #     '''After we finish what was our distance to the goal?'''
    #     return self.smoothL1(final_loc, goal)

    # def final_path_loss(self, path: torch.tensor, gt_waypoints: torch.tensor) -> float:
    #     '''After we finish do path loss over whole path we took and GT'''
    #     return self.mse(path, gt_waypoints)

    def get_loss(
        self, pred_waypoints: torch.tensor, agent_pos: np.ndarray, agent_yaw: float
    ) -> torch.tensor:
        # obstacle_loss = self.obstacle_loss(pred_waypoints)*self.weight_obstacle_loss

        pred_waypoints_path, _ = self.heatmap_to_path(pred_waypoints)

        # pred_waypoints_path_nz = pred_waypoints_path[torch.abs(pred_waypoints_path).sum(dim=1)!=0,...]

        gt_path = self.get_corresponding_waypoints(
            pred_waypoints_path, True, agent_pos, agent_yaw
        )

        gt_waypoints = self.path_to_heatmap(gt_path).detach()
        # gt_waypoint_final = self.get_corresponding_waypoints(pred_waypoints_path_nz[-1,:].reshape(1,2))

        path_loss = self.path_loss(pred_waypoints, gt_waypoints) * self.weight_path_loss
        # position_loss = self.position_loss(pred_waypoints_path_nz[-1,:], gt_waypoint_final)*self.weight_position_loss

        return path_loss + self.get_loss_confidence(pred_waypoints)

    def get_loss_final(
        self,
        final_pred: torch.tensor,
        final_gt: torch.tensor,
        pred_waypoints: torch.tensor,
        agent_pos: np.ndarray,
        agent_yaw: float,
        is_final: bool,
    ) -> torch.tensor:
        weight = (
            is_final * self.weight_final_dist_loss
            + (not is_final) * self.weight_dist_loss
        )

        final_dist_loss = self.path_loss(final_pred, final_gt) * weight

        return final_dist_loss + self.get_loss(pred_waypoints, agent_pos, agent_yaw)

    def get_loss_confidence(self, pred_waypoints: torch.tensor) -> torch.tensor:
        conf = torch.max(pred_waypoints[:])

        loss = (
            1.0
            - torch.clip(conf, 0.0, 1.0)
            + (conf < self.conf_thresh) * torch.square(1.0 - conf)
        )

        return loss * self.weight_confidence_loss
