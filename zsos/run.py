import hydra  # noqa
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat_baselines.run import execute_exp
from omegaconf import DictConfig

# The following imports require habitat to be installed, and despite not being used by
# this script itself, will register several classes and make them discoverable by Hydra.
# This run.py script is expected to only be used when habitat is installed, thus they
# are hidden here instead of in an __init__.py file. noqa is used to suppress the unused
# import and unsorted import warnings by ruff.
import frontier_exploration  # noqa
from habitat import get_config  # noqa
import zsos.obs_transformers.resize  # noqa: F401
from zsos.policy import base_policy, itm_policy, llm_policy  # noqa: F401


@hydra.main(
    version_base=None,
    config_path="../config",
    config_name="experiments/llm_objectnav_hm3d",
)
def main(cfg: DictConfig):
    cfg = patch_config(cfg)
    with read_write(cfg):
        try:
            cfg.habitat.simulator.agents.main_agent.sim_sensors.pop("semantic_sensor")
        except KeyError:
            pass
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()
