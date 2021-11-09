import time
import random

import gym
import torch
from torch import nn
from torch.distributions import Categorical, Normal
import numpy as np
import matplotlib.pyplot as plt

import print_custom as db
import training_args
from logger import *


def normalize(tensor):
    return (tensor - tensor.mean())/(tensor.std() + 1e-7)

def test(env, agent, iter, render=True, maxEpsStep = 5000):
    for _ in range(iter):
        obs = env.reset()
        epsReward = 0
        
        for step in range(maxEpsStep):
            with torch.no_grad():
                # obs = torch.from_numpy(obs).float().unsqueeze(0)
                obs = obs.to(device)
                
                action, _ = agent.sampleAction(obs.unsqueeze(0))
            if render:
                env.render()
            obs, reward, done, _ = env.step(action)
            epsReward += reward
            if done:
                break
        db.printInfo(f"{epsReward=}")

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



def build_mlp(input_dim, output_dim, n_layers, n_hidden,
                activation=nn.Tanh):
    layers = [nn.Linear(input_dim,n_hidden), activation()]
    for _ in range(n_layers):
        layers += [nn.Linear(n_hidden, n_hidden), activation()]
    layers.append(nn.Linear(n_hidden, output_dim))
    return nn.Sequential(*layers)

class ImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        
    def observation(self, obs):
        return torch.tensor(obs.copy()).float().permute(2, 0, 1)

class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, n_hidden=64,
                 activation=nn.Tanh, discrete=True):
        super().__init__()
        self.discrete = discrete
        self.network = build_mlp(input_dim, output_dim, n_layers, n_hidden, activation)
        if not self.discrete:
            self.logstd = nn.Parameter(torch.randn(output_dim))

    def forward(self, input):
        logit = self.network(input) 
        if self.discrete:
            return logit
        else:
            logstd = self.logstd
            return logit, logstd

class ConvNetwork(nn.Module):
    def __init__(self, height, width, channel, output_dim, discrete=True):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channel, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 4, kernel_size=3, stride=1),
            nn.BatchNorm2d(4)
        )
        test_tensor = torch.randn(1, channel, height, width)
        with torch.no_grad():
            test_tensor = self.network(test_tensor).view(-1)

        self.mlp = build_mlp(test_tensor.shape[0], output_dim, 2, 64)
        
        
        self.discrete = discrete
        if not self.discrete:
            self.logstd = nn.Parameter(torch.randn(output_dim))

    def forward(self, input):
        x = self.network(input)
        logit = self.mlp(x.view(input.shape[0], -1))

        if self.discrete:
            return logit
        else:
            logstd = self.logstd
            return logit, logstd

        return 


class Agent():
    def __init__(self, env, actor, critic, training_args) -> None:
        
        # env stuff
        self.env = env
        self.obs = self.env.reset()
        self.is_discrete = isinstance(env.action_space, gym.spaces.Discrete)
        
        # hyper params
        self.discount = training_args.discount
        self.k_epochs = training_args.k_epochs
        self.eps_clip = training_args.clip
        self.lr = training_args.learning_rate
        self.device = training_args.device

        # networks
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr
        )

    def packTrajectory(self, states, actions, rewards, logProbs, isDone):
        
        # trajectory = {"states": torch.stack([torch.from_numpy(s).float() for s in states]).to(self.device),
        #               "actions": torch.stack([torch.tensor(a).float() for a in actions]).to(self.device),
        #               "rewards": rewards,
        #               "isDone": isDone,
        #               "logProbs": torch.tensor(logProbs).float().to(self.device)}

        trajectory = {"states": torch.stack(states).float().to(self.device),
                      "actions": torch.stack([torch.tensor(a).float() for a in actions]).to(self.device),
                      "rewards": rewards,
                      "isDone": isDone,
                      "logProbs": torch.stack(logProbs).float().to(self.device)}
        

        return trajectory

    def computeReturns(self, rewards, isDone, norm=False):
        disReturn = 0
        returns = list()
        
        for rwd, done in zip(rewards[::-1], isDone[::-1]):

            disReturn = self.discount*disReturn* (1-done) + rwd
            returns.insert(0, disReturn)
            
        returns = torch.tensor(returns).float()
        if norm: returns = normalize(returns)
        return returns.to(self.device)
          
    def computeAdvantage(self, returns, values, norm=False):
        advantage = returns - values
        if norm: advantage = normalize(advantage)        
        return advantage.float()
    
    def getValue(self, observation):
        return self.critic(observation)
    
    def getEntropy(self, dist):
        if self.is_discrete:
            return dist.entropy()
        else:
            return dist.entropy().sum(-1)
        
    def getLogProb(self, action, dist):
        if self.is_discrete:
            return dist.log_prob(action)
        else:
            return dist.log_prob(action).sum(-1)

    def sampleAction(self, observation):
        if self.is_discrete:
            dist = Categorical(logits=self.actor(observation))
        else:
            mean, logstd = self.actor(observation)
            dist = Normal(loc=mean, scale=logstd.exp())
        action = dist.sample().to('cpu').numpy()
        return action[0], dist

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
        
        for _ in range(self.k_epochs):

            for n in self.dataIterator(traj, disReturn, batch_size):    
                old_states, old_actions, old_logProbs, returns, = n

                stateVals = self.getValue(old_states)

                stateVals = stateVals.squeeze()
                with torch.no_grad():
                    adv = self.computeAdvantage(returns, stateVals, norm=True)

                # mean, logstd = self.actor(old_states)
                _, dist = self.sampleAction(old_states)
                logProb = self.getLogProb(old_actions, dist)
                entropy = self.getEntropy(dist)

                ratio = torch.exp(logProb - old_logProbs)

                surr1 = ratio*adv
                surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip)*adv
                actor_loss = -torch.min(surr1, surr2) - 0.01*entropy

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

    def sampleTrajectory(self, eps_length, traj_size, continue_from_last=False)-> dict:
        
        states, actions, rewards, isDone, logProbs = \
            list(), list(), list(), list(), list()
        
        def to_buffer(obs, ac, rwd, done, logp):
            states.append(obs)
            actions.append(ac)
            rewards.append(rwd)
            isDone.append(done)
            logProbs.append(logp)
        nStep = 0
        
        eps_step = 0
        if continue_from_last:
            obs = self.obs
        else:
            obs = self.env.reset()

        while nStep < traj_size:
            # obs_tensor = torch.tensor(obs.copy()).float().unsqueeze(0)
            # obs_tensor = obs_tensor.to(self.device)

            obs = obs.to(self.device) 
            action, dist = self.sampleAction(obs.unsqueeze(0))
            ac = torch.tensor(action).float()
            ac = ac.to(self.device)
            logProb = self.getLogProb(ac, dist)

            obs_, reward, done, _ = self.env.step(action)
            nStep +=1
            eps_step+=1
            if eps_step >= eps_length:
                done = True
            to_buffer(obs, action, reward, done, logProb)
            # 
            if done:
                eps_step = 0
                obs = self.env.reset()
            obs = obs_
        self.obs = obs
        return self.packTrajectory(states, actions, rewards, logProbs, isDone)
    
@db.timer
def main(args):

    env = gym.make(args.env_name) 
    test_env = gym.make(args.env_name)

    if args.cnn:
        env = ImgObsWrapper(env)
        test_env = ImgObsWrapper(test_env)
    
    obs_dim = env.observation_space.shape
    db.printInfo(f"{obs_dim}")
    isDiscrete = isinstance(env.action_space, gym.spaces.Discrete)
    if isDiscrete:
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    print(f"====================")
    db.printInfo(f"ENV = {env.unwrapped.spec.id}")
    db.printInfo(f"OBS = {obs_dim}")
    db.printInfo(f"AS = {action_dim}")
    db.printInfo(f"Discrete: {isDiscrete}")
    print(f"====================")

    if args.cnn:
        actor = ConvNetwork(*obs_dim, action_dim, discrete=isDiscrete)
        critic = ConvNetwork(*obs_dim, 1)
    else:
        actor = MLPNetwork(*obs_dim, action_dim,
                        n_layers=args.n_layers, n_hidden=args.n_hidden,
                        discrete=isDiscrete)
        critic = MLPNetwork(*obs_dim, 1,
                            n_layers=args.n_layers, n_hidden=args.n_hidden,
                            discrete=True)
        
    
    agent = Agent(env, actor, critic, args)

    for i in range(args.n_iter):
        with torch.no_grad():
            traj = agent.sampleTrajectory(args.eps_length, args.n_steps, continue_from_last=args.continue_env)
            avgReward = logReward(i, traj['rewards'], traj['isDone'])
        returns = agent.computeReturns(traj['rewards'], traj["isDone"], norm=True)
        agent.updateParams(returns, traj, i, args.batch_size)

        if i % 10 == 0:
            print(f"\n")
            db.printInfo(f"training iter {i} {avgReward=:.3f}")
            test(test_env, agent, 1, False, args.eps_length)
            savePolicy(env.unwrapped.spec.id, agent)
            db.printInfo(f"{G.output_dir=}")
    db.printInfo(f"training iter{i}")
    test(test_env, agent, 5, args.eps_length)
    
if __name__ == '__main__':
    args = training_args.get_args()
    tb_writer = setup_logger(args)
    global device
    device = args.device
    main(args)