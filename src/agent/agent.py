import torch
from abc import ABC, abstractmethod
from src.utils.setup import DEVICE

class Agent(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def train_agent(self):
        pass
    
### inference ###

    def config_policy(self, actor_step, critic_step, batch_size=1, **diffusion_sampling_params):
        # assumes that the savepath has trainer, model, diffusion and dataset configs

        self.critic.load(critic_step)
        self.actor.load(actor_step)
        self.diffusion_model=self.actor.ema_model

        self.diffusion_model.setup_sampling(**diffusion_sampling_params)
        self.batch_size=batch_size

    def policy(self, state):
        """
        Computes an action based on the provided state using a diffusion model.

        This function normalizes the input state, creates a batched version of the state,
        and generates action samples from the diffusion model. The generated actions are then 
        unnormalized before being returned.

        Args:
        state (array-like): The current state from the environment that needs to be processed.

        Returns:
        numpy.ndarray: The first action from the generated samples.
        """
        N=state.shape[0]
        B=self.batch_size

        state = torch.tensor(state, dtype=torch.float32, device=DEVICE)
        state = state.unsqueeze(1).expand(N, B, -1)  # Shape: (N, B, S_dim)
        state = state.reshape(N*B,state.shape[-1]) # B*N_s, S_dim
        samples = self.diffusion_model(state=state).sample.detach() # B*s,(A+1)
        samples = samples.view(N, B, samples.shape[-1])

        actions = samples[:,:, :-1]  # First dimensions for data
        weights = samples[:,:, -1]  # Last dimension for weights
        
        #weights=self.critic.q_target(action=actions, state=state).detach()
        actions=self.select_better(weights, actions)
        return(actions)
    
    def select_better(self, weights, actions):
        argmax_indices = torch.argmax(weights, dim=-1)
        actions = actions[torch.arange(actions.shape[0]), argmax_indices].cpu().numpy()
        return(actions)
    
