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
epsilon_list = [0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001]

runs = []

S = 100
G = .999
# E = 0.0001

# for G in gamma_list:
# for S in states:
for E in epsilon_list:

    P, R = example.forest(S=S, r1=4, r2=2, p=0.1)
    vi = mdp.ValueIteration(P, R, gamma=G, epsilon=E)
    run_stats = vi.run()

    runs.append(run_stats[-1])
    print(vi.policy)

df = pd.DataFrame(runs)

# df.to_csv('vi_runs_gamma.csv')
# df.to_csv('vi_runs_states.csv')
df.to_csv('vi_runs_epsilon.csv')