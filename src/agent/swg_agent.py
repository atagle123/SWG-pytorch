from .agent import Agent
from src.agent.actor import Weighted_actor
from src.agent.critic import IQL_critic


class SelfWeightedAgent(Agent):
    def __init__(self, cfg):
        self.cfg = cfg

        self.actor = Weighted_actor(dataset_params=cfg.dataset, actor_params=cfg.actor)

        self.critic = IQL_critic(dataset_params=cfg.dataset, critic_params=cfg.critic)

    def train_agent(
        self,
        train_critic: bool = True,
        critic_step_load = None,
        train_actor: bool = False,
    ):

        if train_critic:

            self.critic.train(
                n_steps=self.cfg.critic.training.steps,
                log_freq=self.cfg.log_params.log_freq,
                save_freq=self.cfg.log_params.save_freq,
                wandb_log=self.cfg.log_params.wandb_log,
            )

        elif not train_critic and isinstance(critic_step_load, int):
            self.critic.load(step=critic_step_load)

        else:
            print("Error in critic params")

        if train_actor:

            self.actor.dataset.build_weights(
                q_model=self.critic.q_target,
                value_model=self.critic.value_model,
                critic_hyperparam=self.cfg.critic.training.critic_hyperparam,
                weights_function=self.cfg.actor.weight_build.weights_function,
                norm=self.cfg.actor.weight_build.norm,
            )

            self.actor.train(
                n_steps=self.cfg.actor.training.steps,
                log_freq=self.cfg.log_params.log_freq,
                save_freq=self.cfg.log_params.save_freq,
                wandb_log=self.cfg.log_params.wandb_log,
            )
