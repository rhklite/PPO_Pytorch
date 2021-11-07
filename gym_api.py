from inspect import unwrap
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn.modules.activation import Softmax
import gym

import numpy as np
import print_custom as db
import time
from datetime import datetime
from collections import defaultdict
from torch.utils.tensorboard import SummaryWriter
tb_writer = SummaryWriter()
import random

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device='cpu'
def normalize(tensor):
    return (tensor - tensor.mean())/(tensor.std() + 1e-7)

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, n_hidden=64,
                 activation=nn.Tanh):
        super().__init__()
        self.build_mlp(input_dim, output_dim, n_layers, n_hidden, activation)

    def build_mlp(self, input_dim, output_dim, n_layers, n_hidden,
                  activation=nn.Tanh):
        layers = [nn.Linear(input_dim,n_hidden), activation()]
        for _ in range(n_layers):
            layers += [nn.Linear(n_hidden, n_hidden), activation()]
        layers.append(nn.Linear(n_hidden, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, input):
        return self.network(input) 


class Agent():
    def __init__(self, env, actor, critic) -> None:
        self.env = env
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.gamma = 0.99

        self.k_epochs = 4
        self.eps_clip = 0.2

        self.betas = (0.9, 0.999)
        self.lr = 0.002
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
            betas = self.betas
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr,
            betas = self.betas
        )

    def packTrajectory(self, states, actions, rewards, logProbs, isDone):
        
        trajectory = {"states": torch.stack([torch.from_numpy(s).float() for s in states]).to(device),
                      "actions": torch.stack([torch.tensor(a).float() for a in actions]).to(device),
                      "rewards": rewards,
                      "isDone": isDone,
                      "logProbs": torch.tensor(logProbs).float().to(device)}
        return trajectory

    def computeReturns(self, rewards, isDone, norm=False):
        disReturn = 0
        returns = list()
        
        for rwd, done in zip(rewards[::-1], isDone[::-1]):

            disReturn = self.gamma*disReturn* (1-done) + rwd
            returns.insert(0, disReturn)
            
        returns = torch.tensor(returns).float()
        if norm: returns = normalize(returns)
        return returns.to(device)
          
    def computeAdvantage(self, returns, values, norm=False):
        advantage = returns - values
        if norm: advantage = normalize(advantage)        
        return advantage.float()
    
    def getValue(self, observation):
        return self.critic(observation)

    def getLogProb(self, observation, action, entropy=False):
        dist = Categorical(logits=self.actor(observation))
        if entropy:
            return dist.log_prob(action), dist.entropy()
        return dist.log_prob(action)

    def sampleAction(self, observation):
        dist = Categorical(logits=self.actor(observation))
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    
    def dataIterator(self, traj, returns, batch_size):
        n_samples = len(traj['states'])        
        idx = np.arange(0, n_samples, batch_size)
        random.shuffle(idx)
        for n in idx:
            # yield n, n+batch_size
            yield traj['states'][n:n+batch_size], \
                  traj['actions'][n:n+batch_size], \
                  traj['logProbs'][n:n+batch_size], \
                  returns[n:n+batch_size]

    def updateParams(self, disReturn, traj, iteration, batch_size):
        
        for epoch in range(self.k_epochs):

            for n in self.dataIterator(traj, disReturn, batch_size):    
                old_states, old_actions, old_logProbs, returns, = n

                
                # stateVals = self.getValue(traj['states'])
                stateVals = self.getValue(old_states)

                stateVals = stateVals.squeeze()
                with torch.no_grad():
                    adv = self.computeAdvantage(returns, stateVals, norm=True)

                # logProb, entropy = self.getLogProb(traj['states'], traj['actions'], entropy=True)
                # ratio = torch.exp(logProb - traj['logProbs'])
                logProb, entropy = self.getLogProb(old_states, old_actions, entropy=True)
                ratio = torch.exp(logProb - old_logProbs)

                surr1 = ratio*adv
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip)*adv
                actor_loss = -torch.min(surr1, surr2) - 0.005*entropy

                loss = nn.MSELoss()
                critic_loss = loss(stateVals, returns)
                self.actor_optimizer.zero_grad()
                actor_loss.mean().backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.mean().backward()
                self.critic_optimizer.step()


        tb_writer.add_scalar("ActorLoss/train", actor_loss.mean(), iteration, time.time())
        tb_writer.add_scalar("CriticLoss/train", critic_loss.mean(), iteration, time.time())

    def sampleTrajectory(self, maxSteps)-> dict:
        
        states, actions, rewards, isDone, logProbs = \
            list(), list(), list(), list(), list()
        
        def to_buffer(obs, ac, rwd, done, logp):
            states.append(obs)
            actions.append(ac)
            rewards.append(rwd)
            isDone.append(done)
            logProbs.append(logp)
        nStep = 0
        obs = self.env.reset()
        # states.append(obs)
            
        while nStep < maxSteps:
            obs_tensor = torch.from_numpy(obs).float()
            obs_tensor = obs_tensor.to(device)
            action, logProb = self.sampleAction(obs_tensor)
            obs_, reward, done, _ = self.env.step(action)
            nStep +=1
            to_buffer(obs, action, reward, done, logProb)
            if done:
                # states.pop()
                obs = self.env.reset()
                # states.append(obs)
            obs = obs_
        # states.pop()
        
        return self.packTrajectory(states, actions, rewards, logProbs, isDone)
    
def test(env, agent, iter, render=True, maxEpsStep = 5000):

    for _ in range(iter):
        obs = env.reset()
        epsReward = 0
        act = defaultdict(int)
        for step in range(maxEpsStep):
            with torch.no_grad():
                obs = torch.from_numpy(obs).float()
                obs = obs.to(device)
                action, _ = agent.sampleAction(obs)
            act[action] +=1
            if render:
                env.render()
            obs, reward, done, _ = env.step(action)
            epsReward += reward
            if done:
                break
        db.printInfo(f"{epsReward=} {act=}")

def savePolicy(envName, agent, file_name=None):
    if file_name is None:
        file_name = f"{envName}_{datetime.now().strftime('%d%b%Y_%H%M%S')}"
    torch.save(agent, file_name+'.pth')
    return file_name

def logReward(itr, rewards, isDone):
    epsReward = 0
    avgReward = list()

    for r, d in zip(rewards, isDone):
        epsReward +=r
        if d:
            avgReward.append(epsReward)
            epsReward = 0
    if avgReward == []: avgReward = [epsReward]
    tb_writer.add_scalar("Reward", np.mean(avgReward), itr, time.time())
    
    return np.mean(avgReward)

@db.timer
def main():
    # env = gym.make('Breakout-ram-v0')
    env=gym.make('LunarLander-v2')
    env=gym.make('CartPole-v0')
    print(f"====================")
    db.printInfo(f"ENV = {env.unwrapped.spec.id}")
    db.printInfo(f"AS = {env.action_space.n}")
    print(f"====================")
    actor = MLPNetwork(input_dim = env.observation_space.shape[0], 
                       output_dim = env.action_space.n, n_layers=1, n_hidden=64)
    critic = MLPNetwork(env.observation_space.shape[0], 1, n_layers=1, n_hidden=64)

    agent = Agent(env, actor, critic)

    itr = 20000
    n_steps = 1000 # nsteps to collect per iteration
    batch_size = 50
    fileName = None
    for i in range(itr):
        with torch.no_grad():
            traj = agent.sampleTrajectory(n_steps)
            avgReward = logReward(i, traj['rewards'], traj['isDone'])
        returns = agent.computeReturns(traj['rewards'], traj["isDone"], norm=True)
        agent.updateParams(returns, traj, i, batch_size)

        if i % 50 == 0:
            print(f"\n")
            db.printInfo(f"training iter {i} {avgReward=:.3f}")
            test(env, agent, 1, False)
            fileName = savePolicy(env.unwrapped.spec.id, agent, file_name = fileName)
            db.printInfo(f"{fileName=}")
    db.printInfo(f"training iter{i}")
    test(env, agent, 5)
    
if __name__ == '__main__':
    main()