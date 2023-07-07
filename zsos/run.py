import hydra
from habitat.config import read_write
from habitat.config.default import patch_config
from habitat_baselines.run import execute_exp
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../habitat-lab/habitat-baselines/habitat_baselines/config",
    config_name="pointnav/ppo_pointnav_example",
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
