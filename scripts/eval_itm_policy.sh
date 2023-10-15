#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

python -um vlfm.run \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=HabitatITMPolicyV2 \
  habitat.task.lab_sensors.base_explorer.turn_angle=30 \
  habitat_baselines.num_environments=1 \
  habitat_baselines.eval.split=val_50 \
  habitat_baselines.video_dir=itm_aug2 \
  habitat_baselines.eval.video_option='["disk"]'
#  habitat.environment.max_episode_steps=500 \
#  habitat_baselines.test_episode_count=5 \
#  habitat_baselines.eval.video_option='[]'
