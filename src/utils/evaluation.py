import numpy as np
import itertools
from d4rl import get_normalized_score


def evaluate_parallel(policy_fn, envs, env_entry: str, num_episodes: int, sample_params: dict, seed=42) -> dict[str, float]:

    observations = np.array([env.reset() for  env in envs])
    dones = np.zeros(num_episodes, dtype=bool)
    episode_returns = np.zeros(num_episodes)

    # Iterate over environment steps
    while not np.all(dones):
        actions = policy_fn(state=observation)
        
        # Collect rewards and update states
        next_observations = []
        rewards = []
        next_dones = []

        for i, (env, done) in enumerate(zip(envs, dones)):
            observation = observations[i]
            if not done:
                action = actions[i]
                observation, reward, done, _ = env.step(action)
                next_observations.append(observation)
                rewards.append(reward)
                next_dones.append(done)

            else:
                # If the episode is done, we set the reward to 0 and continue with the final state
                next_observations.append(observation)
                rewards.append(0.0)
                next_dones.append(True)

        # Update the states for each environment
        observations = np.array(next_observations)
        dones = np.array(next_dones)
        episode_returns += np.array(rewards)

    scores = get_normalized_score(env_name=env_entry,score=episode_returns)*100 
    scores_mean = np.mean(scores)

    return scores_mean
    

def convert_sample_variants_format(sample_variant: dict) -> list:
    """
    Convert sample variants into a list of dictionaries with all combinations of parameters.
    """
    return [
        dict(zip(sample_variant.keys(), values))
        for values in itertools.product(*sample_variant.values())
    ]


def evaluate_all_variants(agent, envs, env_entry: str, base_sample_params: dict, sample_variant: dict, num_episodes: int) -> dict[str, float]:
    """
    Evaluate the agent for all parameter variants.
    """
    exp_result_dict={}
    params_variant_list = convert_sample_variants_format(sample_variant)
    # Evaluate for each parameter variant
    for param_variants in params_variant_list:
        sample_params = base_sample_params.copy()
        sample_params.update(param_variants)

        agent.config_policy(**sample_params)
        print(f"Evaluating with sample params: {sample_params}")
        eval_info = evaluate_parallel(
            agent.policy, envs, env_entry, num_episodes=num_episodes, sample_params=sample_params
        )

        # Create a unique key for the parameter combination
        param_key = "_".join([f"param_{k}_{v}" for k, v in param_variants.items()])
        exp_result_dict[param_key] = eval_info

    return exp_result_dict