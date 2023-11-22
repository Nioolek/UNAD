import os
import random
import numpy as np
import torch
from torch.backends import cudnn
import torch.distributed as dist


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.enabled = False


def init_dist(args):
    if torch.cuda.is_available():
        cudnn.benchmark = True
        # distributed settings
        if args.launcher == 'none':
            print('Disable distributed.', flush=True)
            rank = 0
            world_size = 1
        else:
            rank = int(os.environ['RANK'])
            num_gpus = torch.cuda.device_count()
            torch.cuda.set_device(rank % num_gpus)
            dist.init_process_group(backend=args.backend)
            world_size = dist.get_world_size()
        print('Using cuda. Rank: {}, world_size: {}'.format(rank, world_size))
        return rank, world_size
    else:
        print('Cuda is not available.')
        return -1, -1
