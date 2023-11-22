import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from unad.loader import LoaderHook
from unad.measure import compute_RMSE
from unad.networks import REDCNNPlus


class SolverPretrain(object):
    """Solver for pretraining UNAD.

    Args:
        cfg (dict): Config dict load from yaml file.
        train_data_loader_hook (LoaderHook): LoaderHook for train dataloader
             to implement Stepwise Patch Increasing Strategy.
        test_data_loader (DataLoader): DataLoader for test dataset.
    """
    def __init__(self,
                 cfg: dict,
                 train_data_loader_hook: LoaderHook,
                 test_data_loader: DataLoader,
                 use_dist: bool = False,
                 rank: int = 0,
                 world_size: int = 1):
        self.cfg = cfg
        self.work_dir = cfg.get('work_dir', './work_dirs/unad_pretrain')
        print('Pretrain results will be saved to : {}'.format(self.work_dir))
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
            print('Create path : {}'.format(self.work_dir))

        # dataloader
        self.train_data_loader_hook = train_data_loader_hook
        self.test_data_loader = test_data_loader

        # train hyperparameters
        self.num_epochs = cfg['epochs']
        self.print_iters = cfg['print_iters']
        self.decay_iters = cfg['decay_iters']
        self.save_iters = cfg['save_iters']
        self.test_interval = cfg['test_interval']
        self.result_fig = cfg['result_fig']

        # visualize and normalize hyperparameters
        self.norm_range_min = cfg['norm_range_min']
        self.norm_range_max = cfg['norm_range_max']
        self.trunc_min = cfg['trunc_min']
        self.trunc_max = cfg['trunc_max']

        # get device
        if cfg.get('device', None):
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(cfg['device'])

        self.use_dist = use_dist
        self.rank = rank
        self.world_size = world_size

        # init model, optimizer, criterion
        self.__init_model()

    def __init_model(self):
        """Initialize UNAD model, optimizer and criterion."""
        # model
        self.UNAD = REDCNNPlus(**self.cfg['UNAD_model'])
        assert self.UNAD.pretrain, 'During pretraining, the `pretrain` in UNAD model must be set to True.'
        self.UNAD.to(self.device)
        self.pretrain_out_ch = self.UNAD.pretrain_out_ch
        assert self.pretrain_out_ch == sum(self.train_data_loader_hook.predict_num), \
            'The pretrain_out_ch must be equal to the sum of predict_num in train_dataloader_hook.'

        if self.use_dist:
            self.UNAD = nn.parallel.DistributedDataParallel(
                self.UNAD,
                device_ids=[torch.cuda.current_device()])

        # loss
        self.criterion = nn.MSELoss(reduction='none')
        self.criterion.to(self.device)
        loss_weight = self.cfg['losses']['loss_weight']
        self.loss_weight = torch.from_numpy(np.array([loss_weight])).detach().view((1, -1, 1, 1)).to(self.device)
        assert len(loss_weight) == self.pretrain_out_ch

        # optimizer
        self.lr = float(self.cfg['lr'])
        self.optimizer = optim.Adam(self.UNAD.parameters(), self.lr)

    def __save_model(self, iter_: int):
        model_path = os.path.join(self.work_dir, 'UNAD_pretrain_{}iter.pth'.format(iter_))
        torch.save(self.UNAD.state_dict(), model_path)
        self.print('Save model to {}'.format(model_path))

    def load_model(self, iter_: int):
        model_path = os.path.join(self.work_dir, 'UNAD_pretrain_{}iter.pth'.format(iter_))
        print('Load model', model_path)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.UNAD.load_state_dict(torch.load(model_path))

    def __lr_decay(self):
        """Decay learning rate by a factor of 0.5."""
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image: np.ndarray):
        """Denormalize image from [0, 1] to [norm_range_min, norm_range_max]."""
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def trunc(self, mat: np.ndarray):
        """Truncate mat to [trunc_min, trunc_max]."""
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def train(self):
        """Train UNAD."""
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(self.num_epochs):
            self.UNAD.train()
            train_data_loader, train_sampler, patch_size = self.train_data_loader_hook.get_data_loader(epoch)
            if train_sampler:
                train_sampler.set_epoch(epoch)
            # x: LDCT img (bs, patch_n, patch_size, patch_size)
            # x_b: LDCT imgs before x  (bs, patch_n, num_before, patch_size, patch_size)
            # x_a: LDCT imgs after  x  (bs, patch_n, num_after, patch_size, patch_size)
            for iter_, (x, x_b, x_a) in enumerate(train_data_loader):
                total_iters += 1
                assert x_b.shape[2] == 3
                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                x_b = x_b.float().to(self.device)
                x_a = x_a.float().to(self.device)
                target = torch.cat([x_b, x_a], dim=2)

                x = x.view(-1, 1, patch_size, patch_size)
                target = target.view(-1, self.pretrain_out_ch, patch_size, patch_size)

                pred = self.UNAD(x)
                loss = (self.criterion(pred, target) * self.loss_weight).mean()

                self.UNAD.zero_grad()
                self.optimizer.zero_grad()

                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

                # print
                if total_iters % self.print_iters == 0:
                    self.print("STEP [{}], EPOCH [{}/{}], ITER [{}/{}] \nLOSS: {:.8f}, TIME: {:.1f}s".format(
                        total_iters,
                        epoch,
                        self.num_epochs,
                        iter_ + 1,
                        len(train_data_loader),
                        loss.item(),
                        time.time() - start_time))
                # learning rate decay
                if total_iters % self.decay_iters == 0:
                    self.__lr_decay()
                # save model
                if (total_iters % self.save_iters == 0) and self.rank <= 0:
                    self.__save_model(total_iters)
                    np.save(os.path.join(self.work_dir, 'loss_pretrain_{}_iter.npy'.format(total_iters)),
                            np.array(train_losses))

                if total_iters % self.test_interval == 0:
                    self.test()
                    self.UNAD.train()

    def test(self):
        """Test UNAD."""
        test_model = self.UNAD
        # Only test on rank 0
        if self.rank > 0:
            return
        self.print('Testing with origin model')
        test_model.eval()
        original_rmse_list = [0. for i in range(self.pretrain_out_ch)]
        pred_rmse_list = [0. for i in range(self.pretrain_out_ch)]

        with torch.no_grad():
            for i, (x, x_b, x_a) in enumerate(self.test_data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                x_b = x_b.float().to(self.device)
                x_a = x_a.float().to(self.device)
                target = torch.cat([x_b, x_a], dim=1)
                pred = test_model(x)

                x = self.trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                target = self.trunc(self.denormalize_(target.view(self.pretrain_out_ch, shape_, shape_).cpu().detach()))
                pred = self.trunc(self.denormalize_(pred.view(self.pretrain_out_ch, shape_, shape_).cpu().detach()))

                for k in range(self.pretrain_out_ch):
                    target_k, pred_k = target[k], pred[k]
                    original_rmse = compute_RMSE(x, target_k)
                    pred_rmse = compute_RMSE(pred_k, target_k)
                    original_rmse_list[k] += original_rmse
                    pred_rmse_list[k] += pred_rmse

            for k in range(self.pretrain_out_ch):
                self.print('slice: {}\nOriginal === \nRMSE avg: {:.4f}'.format(k, original_rmse_list[k] / len(
                    self.test_data_loader)))
                self.print('Predictions === \nRMSE avg: {:.4f}'.format(pred_rmse_list[k] / len(self.test_data_loader)))
                self.print('\n')

    def print(self, msg):
        if self.rank <= 0:
            print(msg)
