import gym
import torch

from training_args import get_args
import print_custom as db
from main import MLPNetwork, Agent, ConvNetwork

device='cuda:0' if torch.cuda.is_available() else 'cpu'

def view_env(env, iter, render=True, maxEpsStep = 5000):
    for _ in range(iter):
        obs = env.reset()
        epsReward = 0
        
        for step in range(maxEpsStep):
            action = env.action_space.sample()
            if render:
                env.render()
            obs, reward, done, _ = env.step(action)
            db.printTensor(obs)
            epsReward += reward
            if done:
                break
        db.printInfo(f"{epsReward=}")

def test(env, agent, iter, render=True, maxEpsStep = 5000):
    for _ in range(iter):
        obs = env.reset()
        epsReward = 0
        
        for step in range(maxEpsStep):
            with torch.no_grad():
                obs = torch.from_numpy(obs).float()
                obs = obs.to(device)
                action, _ = agent.sampleAction(obs)
            if render:
                env.render()
            obs, reward, done, _ = env.step(action)
            epsReward += reward
            if done:
                break
        db.printInfo(f"{epsReward=}")

class ImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        
    def observation(self, obs):
        return torch.tensor(obs.copy()).float().permute(2, 0, 1)

def wrap_test(args):
    
    env = gym.make(args.env_name)
    env = ImgObsWrapper(env)
    obs_dim = env.observation_space.shape
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

    actor = ConvNetwork(*obs_dim, action_dim)
    print(f"{actor}")
    obs = env.reset()
    # obs = torch.tensor(obs.copy()).float()
    db.printTensor(obs.unsqueeze(0))
    action = actor(obs.unsqueeze(0))
    db.printTensor(action)
    # view_env(env, 5, True)
    return
    actor = MLPNetwork(obs_dim, action_dim,
                       n_layers=args.n_layers, n_hidden=args.n_hidden,
                       discrete=isDiscrete)
    critic = MLPNetwork(obs_dim, 1,
                        n_layers=args.n_layers, n_hidden=args.n_hidden,
                        discrete=True)
    agent = Agent(env, actor, critic, args)
    test(env, agent, 5, True)

if __name__ == '__main__':
    args=get_args()
    wrap_test(args)