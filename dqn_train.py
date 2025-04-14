from dqn import QNetwork, make_env, Agent, MaskAgent, learn, Args
import numpy as np

# Value:  0.3754019263817579                                                                                        [64/113]
#   Params:                                                                                                                   
#     lr: 0.0004929820354090957
#     reward_function: fisher
#     theta_state: ml
#     loss_function: l1
#     optimizer: adamw
#     lr_scheduler: CosineAnnealing
#     epsilon_decay_scheduler: exponential
#     target_network_frequency: 8630
#     batch_size: 70
#     buffer_size: 35925
#     end_e: 0.013682892306087196 
#     learning_starts: 57635
#     train_frequency: 1
cli_args = {
    "learning_rate": 0.0004929820354090957,
    "reward_function": "fisher",
    "theta_state": "ml",
    "loss_function": "l1",
    "optimizer": "adamw",
    "lr_scheduler": "CosineAnnealing",
    "e_scheduler": "exponential",
    "target_network_frequency": 8630,
    "batch_size": 70,
    "buffer_size": 35925,
    "end_e": 0.013682892306087196,
    "learning_starts": 57635,
    "train_frequency": 1,
}

args = Args(**cli_args)
N_TRIALS = 10_000  # max num of trials
N_STARTUP_TRIALS = 10  # num of random sampling trials before using TPE sampler
#N_EVALUATIONS = 6  # num of evaluations per trial
N_EVALUATIONS = 1  # num of evaluations per trial
N_TIMESTEPS = 1_000_000  # num of steps per trial
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)  # num of steps before evaluating the agent
N_EVAL_EPISODES = 1000  # num of episodes used for evaluating the agent

ENV_ID = "CAT-v0"
EVAL_EPSILON = 0
CUDA = True
args.cuda = CUDA
args.env_id = ENV_ID
args.total_timesteps = N_TIMESTEPS
args.eval_epsilon = EVAL_EPSILON
args.eval_episodes = N_EVAL_EPISODES
args.eval_frequency = EVAL_FREQ
args.seed = np.random.randint(np.iinfo(np.int32).max)
# Turn off logging in a trial
args.verbose = True
args.track = False

'''
1 - aaron
4 - aaron
7 - tony
10 - tony
13 - yanze
16 - yanze
19 - maher
22 - maher, lawrence
25 - reuben
28 - reuben, lawrence
31 - tony, lawrence
'''
for minutes in [7, 10, 31]:
    args.exp_name = f"dqn_{minutes}_minutes"
    args.time_limit_s = 60 * minutes
    learn(args)
