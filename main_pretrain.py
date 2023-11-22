import argparse
import yaml

from unad.loader import LoaderHook, get_loader
from unad.solver_pretrain import SolverPretrain
from unad.utils import seed_everything, init_dist


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    print('Using pretraining config', cfg)

    # Environment
    seed_everything(1234)
    rank, world_size = init_dist(args)
    use_dist = world_size > 1

    # Train
    train_dataloader_hook = LoaderHook(
        **cfg['train_dataloader_hook'],
        use_dist=use_dist
    )
    test_dataloader, _ = get_loader(**cfg['test_dataloader'])
    solver = SolverPretrain(cfg, train_dataloader_hook, test_dataloader,
                            use_dist=use_dist, rank=rank, world_size=world_size)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='pretrain config file')

    # DDP parameters
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm'],
        default='none',
        help='job launcher')
    parser.add_argument('--backend', default='nccl', type=str)

    args = parser.parse_args()
    main(args)
