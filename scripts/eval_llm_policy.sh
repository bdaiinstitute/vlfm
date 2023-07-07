#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

python -um zsos.run \
  --config-name=experiments/llm_objectnav_hm3d.yaml \
  --config-path ../config \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=LLMPolicy \
  habitat_baselines.rl.ddppo.reset_critic=False \
  habitat.simulator.habitat_sim_v0.allow_sliding=True \
  `# Uncomment (only) one of the next two lines to enable/disable video generation` \
  habitat_baselines.num_environments=1 habitat_baselines.eval.video_option='["disk"]'
#  habitat_baselines.num_environments=1
