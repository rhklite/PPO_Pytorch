import torch
from torch.distributions import Categorical
from main import Agent
from main import MLPNetwork
import argparse
import print_custom as db

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device='cpu'
def loadFile(filepath):
    return torch.load(filepath)

def runEpisode(agent, render=True):
    
    obs = agent.env.reset()
    epsRwd = 0
    while True:
        if render:
            agent.env.render()
        obs_tensor = torch.from_numpy(obs).float().to(device)
        with torch.no_grad():
            action,_ = agent.sampleAction(obs_tensor)
            # logit = agent.actor(obs_tensor)
            # # db.printInfo(f"{logit}")
            # dist = Categorical(logits=logit)
            # db.printInfo(f"{dist.probs}")
        obs, reward, done, _ = agent.env.step(action)
        epsRwd +=reward
        if done:
            return epsRwd    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    args = parser.parse_args()

    trials=5
    agent = torch.load(args.filepath, map_location='cpu')
    avgRwd = 0
    for eps in range(trials):
        epsRwd = runEpisode(agent)
        print(f"Episode {eps} reward {epsRwd}")
        avgRwd +=epsRwd
    print(f"-------------------------------")
    print(f"Avg Reward {avgRwd/trials} in {trials=}")
if __name__ == '__main__':
    main()