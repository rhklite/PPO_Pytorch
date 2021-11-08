import argparse
import json

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '-env', type=str, default='CartPole-v1')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--n_hidden', '-nh', type=int, default=64)
    parser.add_argument('--n_iter', '-n', type=int, default=100,
        help="number of training iterations to run")
    parser.add_argument('--n_steps', '-s', type=int, default=1000,
        help='number of steps to take per training iteration')
    parser.add_argument('--batch_size', '-b', type=int, default=1000,
        help='batch size')
    parser.add_argument('--discount', type=float, default=0.99,
        help="discounnt rate for the return")
    parser.add_argument('--k_epochs', '-ep', type=int, default=4,
        help="perform k epoch updates on each training iteration")
    parser.add_argument('--clip', '-c', type=float, default=0.2,
        help="clip value")
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3,
        help="optimizer learning rate")
    parser.add_argument('--continue_env', '-rtg', action='store_true', default=True,
        help="Don't reset the environment between trajectories")
    parser.add_argument('--dir', '-d', type=str, default='training/')
    parser.add_argument('--load_json',
    help='Load settings from file in json format. Command line options override values in file.')

    return parser

def get_args():
    parser = get_parser()
    args = parser.parse_args()

    if args.load_json:
        with open(args.load_json, 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            args = parser.parse_args(namespace=t_args)
    return args
    # env_name = 'Reacher-v2'
    # env_name = "CartPole-v1"
    # env_name = "LunarLander-v2"
    # env_name = "InvertedPendulum-v2"
    # env_name = "LunarLanderContinuous-v2"
    # env_name= "Hopper-v3"
    # env_name = "BipedalWalker-v3"
    # env_name = "HalfCheetah-v3"
