#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

python -um zsos.run \
  --config-name=objectnav/ddppo_objectnav_hm3d.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=LLMPolicy \
  habitat_baselines.rl.ddppo.reset_critic=False \
  habitat.simulator.habitat_sim_v0.allow_sliding=True \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.video_option='["disk"]'
#  +habitat/task/lab_sensors@habitat.task.lab_sensors.objnav_explorer=objnav_explorer
