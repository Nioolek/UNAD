import argparse
import yaml

from unad.loader import get_loader, LoaderHook
from unad.solver import Solver
from unad.utils import seed_everything, init_dist


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)
    print('Using config: ', cfg)

    # Environment
    seed_everything(1234)
    rank, world_size = init_dist(args)
    use_dist = world_size > 1

    if args.test:
        assert world_size <= 1, 'Test mode does not support distributed.'
        test_dataloader, _ = get_loader(**cfg['test_dataloader'])
        solver = Solver(cfg, None, test_dataloader)
        solver.test(test_iters=args.test_iters)
    else:
        # Get train and test dataloader
        train_dataloader_hook = LoaderHook(
            **cfg['train_dataloader_hook'],
            use_dist=use_dist,
            )
        test_dataloader, _ = get_loader(**cfg['test_dataloader'])
        solver = Solver(cfg, train_dataloader_hook, test_dataloader,
                        use_dist=use_dist, rank=rank, world_size=world_size)
        solver.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='config file')
    parser.add_argument('--test', action='store_true',
                        help='If --test is set, it is the test mode; otherwise, it is the train mode.')
    parser.add_argument('--test_iters', type=int, default=None, help='Indicates which iteration of the model to test.')

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
