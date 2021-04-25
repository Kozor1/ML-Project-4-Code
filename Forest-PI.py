import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
import gym
import hiive.mdptoolbox.example as example
import hiive.mdptoolbox.mdp as mdp
import matplotlib.pyplot as plt

random_seed = 17
np.random.seed(random_seed)

states = np.arange(2,101)

gamma_list = [.9, .99, .999, .9999]

runs = []

# S = 100
# for G in gamma_list:

G = .999
for S in states:

    P, R = example.forest(S=S, r1=4, r2=2, p=0.1)
    vi = mdp.PolicyIteration(P, R, gamma=G)
    run_stats = vi.run()

    runs.append(run_stats[-1])
    # print(vi.policy)

df = pd.DataFrame(runs)

# df.to_csv('pi_runs_gamma.csv')
df.to_csv('pi_runs_states.csv')