#!/usr/bin/env bash
# Copyright [2023] Boston Dynamics AI Institute, Inc.

python -um zsos.run \
  --config-name=objectnav/ddppo_objectnav_hm3d.yaml \
  habitat_baselines.evaluate=True \
  habitat_baselines.eval_ckpt_path_dir=dummy_policy.pth \
  habitat_baselines.load_resume_state_config=False \
  habitat_baselines.rl.policy.name=BasePolicy \
  habitat_baselines.rl.ddppo.reset_critic=False
