import numpy as np

# Environment setup
states = [0, 1, 2, 3, 4]  # 0=start, 4=goal
actions = [0, 1]  # 0=left, 1=right
rewards = [0, 0, 0, 0, 10]  # only state 4 has reward

# Q-table (is a matrix that stores the Q-value for every combination of state and action
#  Q(s,a)  estimating the quality of taking action ‘a’ in state.
Q = np.zeros((len(states), len(actions)))
alpha = 0.5  # learning rate
gamma = 0.9  # discount factor
episodes = 20

# Training
for episode in range(episodes):
    state = 0
    while state != 4:
        action = np.random.choice(actions)
        next_state = max(0, min(state + (1 if action == 1 else -1), 4))
        reward = rewards[next_state]

        # Q-learning update
        Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action]
        )

        state = next_state

print("Trained Q-Table:")
print(Q)