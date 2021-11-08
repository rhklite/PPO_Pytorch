import inspect
import os
import os.path as osp
import json
import time
from datetime import datetime

import torch

import print_custom as db

class G:
    output_dir = None
    output_file = None

def save_hyperparams(output_dir, params):
    with open(osp.join("hyperparams.json"), 'w') as out:
        out.write(json.dumps(params, separators=(',\n','\t:\t'), sort_keys=True))

def configure_output_dir(args):
    """
    Set output directory to d, or to /tmp/somerandomnumber if d is None
    """

    dir = args.dir
    name = args.exp_name
    env_name = args.env_name
    if name == None:
        name = f"{env_name}_{datetime.now().strftime('%d%b%Y_%H%M%S')}"
    G.output_dir = osp.join(dir, name)
    assert not osp.exists(G.output_dir), "Log dir %s already exists! Delete it first or use a different dir"%G.output_dir
    os.makedirs(G.output_dir)
    output_file = open(osp.join(G.output_dir, "log.txt"), 'w')

def savePolicy(envName, agent, file_name=None):
    if file_name is None:
        file_name = f"{envName}_{datetime.now().strftime('%d%b%Y_%H%M%S')}"
    torch.save(agent, os.path.join(G.output_dir, file_name+'.pth'))
    return file_name


def setup_logger(args):

    configure_output_dir(args)
    with open(osp.join(G.output_dir, 'hyperparameters.json'), 'wt') as f:
        json.dump(vars(args), f, indent=4)


