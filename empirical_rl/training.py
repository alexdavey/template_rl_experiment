from rlberry.manager import ExperimentManager
from rlberry.envs import gym_make
from rlberry.agents.stable_baselines import StableBaselinesAgent
from rlberry.seeding import Seeder

import torch
from stable_baselines3 import PPO
from avec_ppo import AVECPPO

seeder = Seeder(42)
env_id = "Ant-v2"
fit_budget = 1e6

# Hyperparams from Table 3
hyperparams = {
    "n_steps": 2048,
    "learning_rate": 2.5e-4,
    "n_epochs": 10,
    "batch_size": 32,
    "gamma": 0.99,
    "policy_kwargs": {"activation_fn": torch.nn.Tanh, "net_arch": {"pi": [64]*2, "vf": [64]*2}},
    "gae_lambda": 0.95,
    "clip_range": 0.2,
}

# The ExperimentManager class is a compact way of experimenting with a deepRL agent.
default_xp = ExperimentManager(
    StableBaselinesAgent,  # The Agent class.
    (gym_make, dict(id=env_id)),  # The Environment to solve.
    fit_budget=fit_budget,  # The number of interactions
    # between the agent and the
    # environment during training.
    init_kwargs={**{"algo_cls": PPO}, **hyperparams},  # Init value for StableBaselinesAgent
    eval_kwargs=dict(eval_horizon=500),  # The number of interactions
    # between the agent and the
    # environment during evaluations.
    n_fit=5,  # The number of agents to train.
    # Usually, it is good to do more
    # than 1 because the training is
    # stochastic.
    seed=seeder,
    agent_name="default_ppo",  # The agent's name.
    output_dir="data_training_default_ppo"
)

avec_xp = ExperimentManager(
    StableBaselinesAgent,  # The Agent class.
    (gym_make, dict(id=env_id)),  # The Environment to solve.
    fit_budget=fit_budget,  # The number of interactions
    # between the agent and the
    # environment during training.
    init_kwargs={**{"algo_cls": AVECPPO}, **hyperparams},  # Init value for StableBaselinesAgent
    eval_kwargs=dict(eval_horizon=500),  # The number of interactions
    # between the agent and the
    # environment during evaluations.
    n_fit=5,  # The number of agents to train.
    # Usually, it is good to do more
    # than 1 because the training is
    # stochastic.
    seed=seeder,
    agent_name="avec_ppo",  # The agent's name.
    output_dir="data_training_avec_ppo"
)

default_xp.fit(), avec_xp.fit()



# FOR TESTING PURPOSES
from rlberry.manager import plot_writer_data

_ = plot_writer_data([default_xp, avec_xp],
    tag="rollout/ep_rew_mean",
    title="Training Episode Cumulative Rewards",
    show=False,
    savefig_fname="rewards"
)

_ = plot_writer_data([default_xp, avec_xp],
    tag="train/explained_variance",
    title="Training Explained Variance",
    show=False,
    savefig_fname="explained_variance"
)

_ = plot_writer_data([default_xp, avec_xp],
    tag="train/value_loss",
    title="Training Value Loss",
    show=False,
    savefig_fname="value_loss"
)

from rlberry.manager import evaluate_agents
import matplotlib.pyplot as plt

# Comparing means
_ = evaluate_agents(
    [default_xp, avec_xp], n_simulations=50,show=False,
)  # Evaluate the trained agent on
plt.savefig("evaluations")
