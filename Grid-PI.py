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

runs = []

# S = 10
# for G in gamma_list:

G = .999
for S in states:

    random_map = generate_random_map(size=S, p=0.9)
    P, R = example.openai(env_name="FrozenLake-v0", desc=random_map)
    vi = mdp.PolicyIteration(P, R, gamma=G)
    run_stats = vi.run()

    runs.append(run_stats[-1])
    # print(vi.policy)

df = pd.DataFrame(runs)

# df.to_csv('grid_pi_runs_gamma.csv')
df.to_csv('grid_pi_runs_states.csv')