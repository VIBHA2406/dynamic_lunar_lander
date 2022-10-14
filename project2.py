import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = 50
n_episodes = 2000
steps_per_episode = 1000
alpha = 0.0001
epsilon_start = 1
epsilon_end = 0.
epsilon_decay = 0.99
gamma = 0.99
buffer_size = 100000
Update_freq = 4
mini_batch = 32


class dqn_network(nn.Module):
    def __init__(self, state_size, action_size, seed):
        super(dqn_network, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Sequential(nn.Linear(state_size, 256), nn.Dropout(0.5))
        self.fc2 = nn.Linear(256, 128)
        #self.fc_batch = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(128, action_size)

#add a layer
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return (self.fc4(x))


class Model:
    #Initializing the RL Agent
    def __init__(self,state_space, action_space, seed):
        self.state_space = state_space
        self.action_space = action_space
        self.seed = seed

        #Defining the Q_Network
        self.q_network = dqn_network(state_space, action_space,seed).to(device)
        self.q_target = dqn_network(state_space, action_space,seed).to(device)
        #self.q_target.load_state_dict(self.q_network.state_dict())
        #self.q_target.eval()

        #Defining the Optimizer for the network
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=alpha)

        #Defining the Experience Replay Buffer:
        self.memory = Experience_Replay_Buffer(action_space, buffer_size, mini_batch, seed)
        self.steps = 0

    def take_action(self, s, epsilon):
        state = torch.from_numpy(s).float().unsqueeze(0).to(device)
        action = self.epsilon_greedy(state,epsilon)
        return action

    def epsilon_greedy(self, state, epsilon):
        #self.q_network.train()
        if np.random.random() < epsilon:
            action = np.random.choice(np.arange(self.action_space))
        else:
            with torch.no_grad():
                action_val = self.q_network(state)
            action = np.argmax(action_val.cpu().data.numpy())
        return action

    def learn(self, experience_t, gamma):

        states, actions, rewards, next_states, done_t = experience_t
        next_target = self.q_target(next_states).detach().max(1)[0].unsqueeze(1)
        Current_Q = rewards + (gamma*next_target*(1-done_t))
        # self.q_target.train()
        Expected_Q = self.q_network(states).gather(1,actions)
        #Computing loss function and evaluating the learning
        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(Expected_Q, Current_Q)
        #print(loss.detach().numpy())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_graphs(self, scores, plot1):
        if plot1:
            plt.plot(scores)
            plt.ylabel('Total Reward', fontsize=16)
            plt.xlabel('Episodes', fontsize=16)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            #plt.plot(np.arange(len(scores)), scores)
            plt.title('Reward per episode', fontsize=20)
            plt.grid(True)
            plt.savefig('Plot_1')
            plt.clf()
class Experience_Replay_Buffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def sample(self):
        experiences = random.sample(self.memory, k = self.batch_size)
        #print(experiences)
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        #print(states)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        #print(actions)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        #print(rewards)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        #print(next_states)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def __len__(self):
        return len(self.memory)

if __name__ == "__main__":
    
    env = gym.make('LunarLander-v2').unwrapped
    np.random.seed(seed)
    env.seed(seed)
    action_space_size = 4
    state_space_size = 8
    agent = Model(state_space_size, action_space_size, seed)
    total_reward = []
    reward_window = deque(maxlen = 100)
    #img = plt.imshow(env.render(mode='rgb_array'))
    epsilon = epsilon_start
    for episode in range(n_episodes):
        s = env.reset()
        t = 0
        score = 0
        while t < steps_per_episode:
            t += 1
            a = agent.take_action(s, epsilon)
            new_state, reward, done, info = env.step(a)
            agent.memory.add(s, a, reward, new_state,done)
            if done:
                #print(t)
                break
            #TODO try priortized sweeping
            if len(agent.memory) > mini_batch:
                experiences = agent.memory.sample()
                agent.learn(experiences, gamma)
                #agent.update_q()
                agent.steps = (agent.steps + 1) % Update_freq
                if agent.steps == 0:
                    agent.q_target.load_state_dict(agent.q_network.state_dict())

            s = new_state
            score += reward
        reward_window.append(score)
        if epsilon > epsilon_end:
            epsilon *= epsilon_decay
        if score >= 200:
            print("This episode solved it!")
        #agent.steps = (agent.steps + 1) % Update_freq

        total_reward.append(score)
        #print('\n\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(reward_window)), end="")
        if episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(episode, np.mean(reward_window)))
        if np.mean(reward_window) >= 115.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(episode - 100,
                                                                                         np.mean(reward_window)))
            torch.save(agent.q_network.state_dict(), 'checkpoint.pth')

    #graph modifications pending
    # plot the scores
    agent.plot_graphs(total_reward, plot1=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(total_reward)), total_reward)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.show()

