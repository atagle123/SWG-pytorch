import os
import hydra
import logging
from src.utils.setup import set_seed, DEVICE
from src.agent.swg_agent import SelfWeightedAgent

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
config_path = "../configs/D4RL"


@hydra.main(config_path=config_path, config_name="config", version_base="1.3")
def main(cfg):
    logger.info(f"Starting experiment with config: {cfg}")
    logger.info(f"Using DEVICE: {DEVICE}")

    set_seed(cfg.seed)

    os.makedirs(cfg.actor.actor_savepath, exist_ok=True)
    os.makedirs(cfg.critic.critic_savepath, exist_ok=True)

    agent = hydra.utils.instantiate(cfg.method.agent, cfg)
    agent.train_agent(train_critic=True, critic_step_load=1000000, train_actor=True)


if __name__ == "__main__":
    main()
