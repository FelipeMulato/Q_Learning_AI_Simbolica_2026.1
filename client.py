import connection as env
import numpy as np

#Utils Functions
def decay_schedule(init_value,min_value,
                    decay_ratio, max_steps,
                    log_start = -2, log_base =10):
    decay_steps = int(max_steps*decay_ratio)
    rem_steps = max_steps-decay_steps

    values = np.logspace(log_start, 0, decay_steps,
                         base = log_base, endpoint =True)
    values = (values-values.min())

    values = (init_value-min_value)*values +min_value

    values = np.pad(values,(0, rem_steps), 'edge')

    return values

def select_action(Q,state, epsilon):
        if np.random.random() > epsilon:
            return np.argmax(Q[state])
        else:
            return np.random.randint(len(Q[state]))





#Hyperaments
gamma=1.0
init_alpha=0.5
min_alpha=0.01
alpha_decay_ratio=0.3
init_epsilon=1.0
min_epsilon=0.1
epsilon_decay_ratio=0.9
n_episodes=500
max_steps=30
state_size = 24*4
action_size = 3

actions = ["left","right","jump"]

Q = np.zeros((state_size,action_size))

alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)

for e in range(0, n_episodes):
    state_d = 0
    for step in range(max_steps):

        state = env.connect(2037)

        action  = select_action(Q,state_d,epsilons[e])
        action_s = actions[action]

        new_state, reward = env.get_state_reward(state, action_s)

        new_state_d = int(new_state[0:7],2) + 24*int(new_state[7:],2)

        #Acabou o Episodio 
        if(new_state_d==0):
            Q[state_d][action] = Q[state_d][action] + alphas[e]*(reward)
            break
             
        Q[state_d][action] = Q[state_d][action] + alphas[e]*(reward + gamma*(np.max(Q[new_state_d]) - Q[state_d][action]))

        state_d = new_state_d
    



