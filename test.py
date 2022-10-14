import gym
from project2 import Model
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

seed = 50
if __name__ == "__main__":
    env = gym.make('LunarLander-v2')
    np.random.seed(seed)
    env.seed(seed)
    action_space_size = 4
    state_space_size = 8
    agent = Model(state_space_size, action_space_size, seed)
    total_reward = []
    
    agent.q_target.load_state_dict(torch.load('checkpoint.pth'))

    for i in range(100):
        state = env.reset()
        score = 0
        #img = plt.imshow(env.render(mode='rgb_array'))
        for j in range(1000):
            action = agent.take_action(state, epsilon = 0.5)
            #img.set_data(env.render(mode='rgb_array'))
            state, reward, done, _ = env.step(action)
            if done:
                print(score + reward)
                break
            score += reward
            total_reward.append(score)
        env.close()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_reward)), total_reward)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title('Reward per episode', fontsize=20)
    plt.grid(True)
    plt.savefig('Temp_Test')
