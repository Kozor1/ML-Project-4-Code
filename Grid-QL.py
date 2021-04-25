import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import gym
import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
import matplotlib.pyplot as plt
from gym.envs.toy_text.frozen_lake import generate_random_map

random_seed = 17
np.random.seed(random_seed)

states = np.arange(2,51)
gamma_list = [.9, .99, .999, .9999]
states = np.arange(2,101)
alphas = np.arange(0.5,1.0,0.1)
alpha_decays = [.9, .99, .999, .9999]

runs = []

G = .999
S = 50
# for G in gamma_list:
# for S in states:
for alpha in alphas:
    for alpha_dec in alpha_decays:

        random_map = generate_random_map(size=S, p=0.9)
        P, R = example.openai(env_name="FrozenLake-v0", desc=random_map)
        vi = mdp.QLearning(P, R, gamma=G, n_iter=100000, epsilon=0.0001, alpha=alpha, alpha_decay=alpha_dec)
        run_stats = vi.run()

        runs.append(run_stats[-1])
        # print(vi.policy)

df = pd.DataFrame(runs)

# df.to_csv('grid_ql_runs_gamma.csv')
#df.to_csv('grid_ql_runs_states.csv')
df.to_csv('grid_ql_runs_alpha.csv')