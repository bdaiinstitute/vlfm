# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf


def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


@baseline_registry.register_trainer(name="vln")
class VLNTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # get args
        args = self.config.habitat_baselines.rl.policy.options

        log_dir = args.logging.log_dir
        os.makedirs(log_dir, exist_ok=True)

        ORACLE_STOP = args.enable_oracle_stop
        LOG_SUCCES_IF_ORACLE_STOP = args.logging.enable_log_success_if_oracle_stop
        LOG_THRESH = args.logging.enable_log_success_thresh

        # set-up failure analysis
        os.makedirs(log_dir + args.logging.analysis_save_location, exist_ok=True)
        file_success = open(args.logging.analysis_save_location + "successes.txt", "w")
        file_fail = open(args.logging.analysis_save_location + "failures.txt", "w")

        file_log = open(log_dir + "logging_info.txt", "w")

        if ORACLE_STOP or LOG_SUCCES_IF_ORACLE_STOP:
            self.should_stop = False

        if LOG_THRESH:
            self.thresh_dict: Dict[Tuple[float, float], List[int]] = {}
            for i in np.linspace(-1.0, 1.5, 26):
                for j in np.linspace(-20.0, 40.0, 61):
                    self.thresh_dict[(i, j)] = []
            best_thresh = None

        # gt_path_for_viz = None

        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        self._init_envs(config, is_eval=True)

        self._agent = self._create_agent(None)
        action_shape, discrete_actions = get_action_space_info(
            self._agent.policy_action_space
        )

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset()

        # Split off instructions so it doesn't get messed up in the batching
        instructions = []
        for j in range(len(observations)):
            instructions += [observations[j]["instruction"]["text"]]
            del observations[j]["instruction"]

        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = (
            {}
        )  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [
            [] for _ in range(self.config.habitat_baselines.num_environments)
        ]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(
                log_dir + self.config.habitat_baselines.video_dir, exist_ok=True
            )

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert (
            number_of_eval_episodes > 0
        ), "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        if LOG_SUCCES_IF_ORACLE_STOP:
            num_os_successes = 0

        hab_vis = HabitatVis()
        while (
            len(stats_episodes) < (number_of_eval_episodes * evals_per_ep)
            and self.envs.num_envs > 0
        ):
            current_episodes_info = self.envs.current_episodes()

            with inference_mode():
                gt_path_for_viz, gt_path_for_viz_wc = self.envs.call(["get_gt_path"])[0]
                gt_path_for_viz = gt_path_for_viz[:, :2]
                gt_path_for_viz_wc = gt_path_for_viz_wc[:, :2]
                self._agent.actor_critic.set_envs(self.envs)
                self._agent.actor_critic.set_instruction(instructions[0])
                self._agent.actor_critic.set_gt_path_for_viz(
                    gt_path_for_viz, gt_path_for_viz_wc
                )
                action_data = self._agent.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )
                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # If the file doesn't exist, create it
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = (
                                action_data.rnn_hidden_states[i]
                            )
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            if ORACLE_STOP:
                if self.should_stop:
                    step_data = [0]
                elif step_data[0] == 0:
                    step_data = [np.random.randint(1, 4)]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])

            if ORACLE_STOP:
                self.should_stop = (
                    infos[0]["distance_to_goal"]
                ) <= self.config.habitat.task.measurements.success.success_distance

            if LOG_SUCCES_IF_ORACLE_STOP:
                if (
                    infos[0]["distance_to_goal"]
                ) <= self.config.habitat.task.measurements.success.success_distance:
                    self.should_stop = True
                    print("WITHIN GOAL DIST! ", infos[0]["distance_to_goal"])

            if LOG_THRESH:
                thresh = self._agent.actor_critic._path_selector.get_last_thresh()
                if thresh is not None:
                    in_dist = (
                        (infos[0]["distance_to_goal"])
                        <= self.config.habitat.task.measurements.success.success_distance
                    )

                    best_thresh = None
                    best_val = 0

                    for k in self.thresh_dict.keys():
                        if in_dist:
                            if thresh[0] <= k[0] and thresh[1] <= k[1]:
                                if num_total not in self.thresh_dict[k]:
                                    self.thresh_dict[k] += [num_total]

                        if len(self.thresh_dict[k]) > best_val:
                            best_thresh = k
                            best_val = len(self.thresh_dict[k])

                    if (
                        best_thresh is not None
                    ) and best_thresh in self.thresh_dict.keys():
                        print(
                            f"BEST THRESH SO FAR: {best_thresh}, gives"
                            f" {len(self.thresh_dict[best_thresh])}/{num_total+1} ="
                            f" {len(self.thresh_dict[best_thresh])/(num_total+1)}"
                        )
                    else:
                        print("NO BEST THRESH")

            # gt_path_for_viz = np.array(infos[0]["gt_path_vln"])
            # gt_path_for_viz = gt_path_for_viz[:, :2]

            # Split off instructions
            instructions_curr = instructions.copy()

            instructions = []
            for j in range(len(observations)):
                instructions += [observations[j]["instruction"]["text"]]
                del observations[j]["instruction"]

            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                if len(self.config.habitat_baselines.eval.video_option) > 0:
                    hab_vis.collect_data(batch, infos, action_data.policy_info)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                        file_success.write(instructions_curr[0])
                        file_success.write("\n")
                        file_success.flush()
                    else:
                        file_fail.write(instructions_curr[0])
                        file_fail.write("\n")
                        file_fail.flush()
                    num_total += 1
                    print("\n", instructions_curr[0])
                    print(
                        f"Success rate: {num_successes / num_total * 100:.2f}% "
                        f"({num_successes} out of {num_total})"
                    )

                    file_log.write(
                        "Success rate:"
                        f" {num_successes / num_total * 100:.2f} ({num_successes} out"
                        f" of {num_total})"
                    )
                    file_log.write("\n")

                    if LOG_SUCCES_IF_ORACLE_STOP:
                        if self.should_stop:
                            num_os_successes += 1
                        print(
                            "Success rate with OS:"
                            f" {num_os_successes / num_total * 100:.2f}%"
                            f" ({num_os_successes} out of {num_total})"
                        )

                        file_log.write(
                            "Success rate with OS:"
                            f" {num_os_successes / num_total * 100:.2f} ({num_os_successes} out"
                            f" of {num_total})"
                        )
                        file_log.write("\n")

                    if LOG_THRESH:
                        if (
                            best_thresh is not None
                        ) and best_thresh in self.thresh_dict.keys():
                            print(
                                f"BEST THRESH SO FAR: {best_thresh}, gives"
                                f" {len(self.thresh_dict[best_thresh])}/{num_total} ="
                                f" {len(self.thresh_dict[best_thresh])/num_total}"
                            )
                            file_log.write(
                                f"BEST THRESH SO FAR: {best_thresh}, gives"
                                f" {len(self.thresh_dict[best_thresh])}/{num_total} ="
                                f" {len(self.thresh_dict[best_thresh])/num_total}"
                            )
                            file_log.write("\n")
                            close0 = 0.5
                            close1 = 5.0
                            # print("CLOSE THRESHOLD VALS")
                            file_log.write("CLOSE THRESHOLD VALS")
                            file_log.write("\n")
                            for k in self.thresh_dict.keys():
                                if (
                                    np.abs(k[0] - best_thresh[0]) < close0
                                    and np.abs(k[1] - best_thresh[1]) < close1
                                ):
                                    # print("* ", k, len(self.thresh_dict[k]))
                                    file_log.write(f"* {k}: {len(self.thresh_dict[k])}")
                                    file_log.write("\n")

                    file_log.flush()

                    from vlfm.utils.episode_stats_logger import (
                        log_episode_stats,
                    )

                    try:
                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )
                    except Exception:
                        failure_cause = "Unknown"

                    if len(self.config.habitat_baselines.eval.video_option) > 0:
                        rgb_frames[i] = hab_vis.flush_frames(failure_cause)
                        generate_video(
                            video_option=self.config.habitat_baselines.eval.video_option,
                            video_dir=log_dir + self.config.habitat_baselines.video_dir,
                            images=rgb_frames[i],
                            episode_id=current_episodes_info[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=extract_scalars_from_info(infos[i]),
                            fps=self.config.habitat_baselines.video_fps,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                        )

                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

                    if ORACLE_STOP or LOG_SUCCES_IF_ORACLE_STOP:
                        self.should_stop = False

                    # gt_path_for_viz = None

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        file_success.close()
        file_fail.close()
        file_log.close()

        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean(
                [v[stat_key] for v in stats_episodes.values()]
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
