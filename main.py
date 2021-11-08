import time
import random
import numpy as np

import gym
import torch
from torch import nn
from torch.distributions import Categorical, Normal
from torch.utils.tensorboard import SummaryWriter

import print_custom as db
import training_args
from logger import *

tb_writer = SummaryWriter()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device='cpu'
def normalize(tensor):
    return (tensor - tensor.mean())/(tensor.std() + 1e-7)

def test(env, agent, iter, render=True, maxEpsStep = 5000):

    for _ in range(iter):
        obs = env.reset()
        epsReward = 0
        
        for step in range(maxEpsStep):
            with torch.no_grad():
                obs = torch.from_numpy(obs).float()
                obs = obs.to(device)
                action, _ = agent.sampleAction(obs)
            action = env.action_space.sample()
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


class MLPNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers=1, n_hidden=64,
                 activation=nn.Tanh, discrete=True):
        super().__init__()
        self.discrete = discrete
        self.build_mlp(input_dim, output_dim, n_layers, n_hidden, activation)

    def build_mlp(self, input_dim, output_dim, n_layers, n_hidden,
                  activation=nn.Tanh):
        layers = [nn.Linear(input_dim,n_hidden), activation()]
        for _ in range(n_layers):
            layers += [nn.Linear(n_hidden, n_hidden), activation()]
        layers.append(nn.Linear(n_hidden, output_dim))
        self.network = nn.Sequential(*layers)

        if not self.discrete:
            self.logstd = nn.Parameter(torch.randn(output_dim))

    def forward(self, input):
        if self.discrete:
            return self.network(input) 
        else:
            mean = self.network(input)
            logstd = self.logstd
            return mean, logstd

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

        # networks
        self.actor = actor.to(device)
        self.critic = critic.to(device)

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.lr,
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.lr
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

            disReturn = self.discount*disReturn* (1-done) + rwd
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
        action = dist.sample()
        return action.to('cpu').numpy(), dist

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

    def sampleTrajectory(self, maxSteps, continue_from_last=False)-> dict:
        
        states, actions, rewards, isDone, logProbs = \
            list(), list(), list(), list(), list()
        
        def to_buffer(obs, ac, rwd, done, logp):
            states.append(obs)
            actions.append(ac)
            rewards.append(rwd)
            isDone.append(done)
            logProbs.append(logp)
        nStep = 0
        
        if continue_from_last:
            obs = self.obs
        else:
            obs = self.env.reset()
            
        while nStep < maxSteps:
            obs_tensor = torch.from_numpy(obs).float()
            obs_tensor = obs_tensor.to(device)
            action, dist = self.sampleAction(obs_tensor)

            ac = torch.tensor(action).float()
            ac = ac.to(device)
            logProb = self.getLogProb(ac, dist)

            obs_, reward, done, _ = self.env.step(action)
            nStep +=1
            to_buffer(obs, action, reward, done, logProb)
            if done:
                obs = self.env.reset()
            obs = obs_
        self.obs = obs
        return self.packTrajectory(states, actions, rewards, logProbs, isDone)
    
@db.timer
def main(args):
    env = gym.make(args.env_name)
    test_env = gym.make(args.env_name)
    
    obs_dim = env.observation_space.shape[0]
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
    actor = MLPNetwork(obs_dim, action_dim,
                       n_layers=args.n_layers, n_hidden=args.n_hidden,
                       discrete=isDiscrete)
    critic = MLPNetwork(obs_dim, 1,
                        n_layers=args.n_layers, n_hidden=args.n_hidden,
                        discrete=True)
    agent = Agent(env, actor, critic, args)

    fileName = None
    for i in range(args.n_iter):
        with torch.no_grad():
            traj = agent.sampleTrajectory(args.n_steps, continue_from_last=args.continue_env)
            avgReward = logReward(i, traj['rewards'], traj['isDone'])
        returns = agent.computeReturns(traj['rewards'], traj["isDone"], norm=True)
        agent.updateParams(returns, traj, i, args.batch_size)

        if i % 10 == 0:
            print(f"\n")
            db.printInfo(f"training iter {i} {avgReward=:.3f}")
            test(test_env, agent, 1, False)
            fileName = savePolicy(env.unwrapped.spec.id, agent, file_name = fileName)
            db.printInfo(f"{fileName=}")
    db.printInfo(f"training iter{i}")
    test(test_env, agent, 5)
    
if __name__ == '__main__':
    args = training_args.get_args()
    setup_logger(args)
    main(args)