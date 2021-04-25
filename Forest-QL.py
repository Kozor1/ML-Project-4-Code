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
alphas = np.arange(0.5,1.0,0.1)
alpha_decays = [.9, .99, .999, .9999]


runs = []

S = 100
G = .999

# for G in gamma_list:
# for S in states:
for alpha in alphas:
    for alpha_dec in alpha_decays:

        P, R = example.forest(S=S, r1=4, r2=2, p=0.1)
        vi = mdp.QLearning(P, R, gamma=G, n_iter=100000, alpha=alpha, alpha_decay=alpha_dec)
        run_stats = vi.run()

        runs.append(run_stats[-1])
        #print(vi.policy)

df = pd.DataFrame(runs)

# df.to_csv('ql_runs_gamma.csv')
#df.to_csv('ql_runs_states.csv')
df.to_csv('ql_runs_alpha.csv')