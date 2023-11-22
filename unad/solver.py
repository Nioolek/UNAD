import os
import time
from collections import OrderedDict

import numpy as np
import matplotlib
from torch.utils.data import DataLoader

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from unad.networks import REDCNNPlus
from unad.measure import compute_measure
from unad.loader import LoaderHook


class Solver(object):
    """Solver for training and testing UNAD.

    Args:
        cfg (dict): Config dict load from yaml file.
        train_data_loader_hook (LoaderHook): LoaderHook for train dataloader
             to implement Stepwise Patch Increasing Strategy.
        test_data_loader (DataLoader): DataLoader for test dataset.
        use_dist (bool): Whether to use distributed training.
    """

    def __init__(self,
                 cfg: dict,
                 train_data_loader_hook: LoaderHook,
                 test_data_loader: DataLoader,
                 use_dist: bool = False,
                 rank: int = 0,
                 world_size: int = 1):
        self.cfg = cfg
        self.work_dir = cfg.get('work_dir', './work_dirs/unad')
        print('Experiment results will be saved to : {}'.format(self.work_dir))
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
            print('Create path : {}'.format(self.work_dir))
        self.fig_dir = os.path.join(self.work_dir, 'fig')
        if not os.path.exists(self.fig_dir):
            os.makedirs(self.fig_dir)

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

        self.pretrain_path = self.cfg.get('pretrain_path', None)

        self.use_dist = use_dist
        self.rank = rank
        self.world_size = world_size

        # init model, optimizer, criterion
        self.__init_model()

    def __init_model(self):
        """Initialize UNAD model, optimizer and criterion."""
        # model
        self.UNAD = REDCNNPlus(**self.cfg['UNAD_model'])
        if self.pretrain_path:
            state_dict = torch.load(self.pretrain_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            del new_state_dict['output.weight']
            del new_state_dict['output.bias']
            self.UNAD.load_state_dict(new_state_dict, strict=False)
            print('Load pretrained model: ', self.pretrain_path)
        self.UNAD.to(self.device)
        self.reg_max = self.UNAD.reg_max
        self.hu_interval = self.UNAD.hu_interval
        self.y_0 = self.UNAD.y_0

        if self.use_dist:
            self.UNAD = nn.parallel.DistributedDataParallel(
                self.UNAD,
                device_ids=[torch.cuda.current_device()])

        # loss
        self.criterion = nn.MSELoss()
        self.criterion.to(self.device)
        # Distribution Focal Loss
        loss_cfg = self.cfg['losses']
        self.use_dflloss = loss_cfg.get('use_dflloss', False)
        if self.use_dflloss:
            self.dfl_lossweight = loss_cfg.get('dfl_lossweight', 0.2)
            self.dfl_eps = loss_cfg.get('dfl_eps', 1e-5)

        # optimizer
        self.lr = float(self.cfg['lr'])
        self.optimizer = optim.Adam(self.UNAD.parameters(), self.lr)

    def __save_model(self, iter_):
        model_path = os.path.join(self.work_dir, 'UNAD_{}iter.pth'.format(iter_))
        torch.save(self.UNAD.state_dict(), model_path)
        self.print('Save model to {}'.format(model_path))

    def load_model(self, iter_):
        model_path = os.path.join(self.work_dir, 'UNAD_{}iter.pth'.format(iter_))
        print('Load model', model_path)
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        self.UNAD.load_state_dict(new_state_dict)

    def __lr_decay(self):
        """Decay learning rate by a factor of 0.5."""
        lr = self.lr * 0.5
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def denormalize_(self, image):
        """Denormalize image from [0, 1] to [norm_range_min, norm_range_max]."""
        image = image * (self.norm_range_max - self.norm_range_min) + self.norm_range_min
        return image

    def __trunc(self, mat):
        """Truncate mat to [trunc_min, trunc_max]."""
        mat[mat <= self.trunc_min] = self.trunc_min
        mat[mat >= self.trunc_max] = self.trunc_max
        return mat

    def save_fig(self, x, y, pred, original_result, pred_result, path):
        x, y, pred = x.numpy(), y.numpy(), pred.numpy()
        f, ax = plt.subplots(1, 3, figsize=(30, 10))
        save_name = os.path.basename(path[0])

        ax[0].imshow(x, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[0].set_title('Quarter-dose', fontsize=30)
        ax[0].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(original_result[0],
                                                                           original_result[1],
                                                                           original_result[2]), fontsize=20)
        ax[1].imshow(pred, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[1].set_title('Result', fontsize=30)
        ax[1].set_xlabel("PSNR: {:.4f}\nSSIM: {:.4f}\nRMSE: {:.4f}".format(pred_result[0],
                                                                           pred_result[1],
                                                                           pred_result[2]), fontsize=20)
        ax[2].imshow(y, cmap=plt.cm.gray, vmin=self.trunc_min, vmax=self.trunc_max)
        ax[2].set_title('Full-dose', fontsize=30)

        f.savefig(os.path.join(self.fig_dir, save_name + '.png'))
        plt.close()

    def dfl_loss(self, out_dist, x, y):
        """Implentation of Distribution Focal Loss.
        The code is referenced from <https://arxiv.org/pdf/2006.04388.pdf>_.
        """
        out_dist = out_dist.reshape((-1, self.reg_max + 1))

        # denormalize label
        label = (y - x) * (self.norm_range_max - self.norm_range_min)
        # convert label to distribution
        label = (label - self.y_0) / self.hu_interval
        # clamp
        label = label.clamp(min=self.dfl_eps, max=self.reg_max - self.dfl_eps).reshape(-1)

        # Distribution Focal Loss
        dis_left = label.long()
        dis_right = dis_left + 1
        weight_left = dis_right.float() - label
        weight_right = label - dis_left.float()
        loss_dfl = F.cross_entropy(out_dist, dis_left, reduction='none') * weight_left \
                   + F.cross_entropy(out_dist, dis_right, reduction='none') * weight_right
        return loss_dfl.mean()

    def train(self):
        """Train UNAD."""
        train_losses = []
        total_iters = 0
        start_time = time.time()
        for epoch in range(self.num_epochs):
            self.UNAD.train(True)
            train_data_loader, train_sampler, patch_size = self.train_data_loader_hook.get_data_loader(epoch)
            if train_sampler:
                train_sampler.set_epoch(epoch)

            # x: LDCT img (bs, patch_n, patch_size, patch_size)
            # y: NDCT img (bs, patch_n, patch_size, patch_size)
            for iter_, (x, y) in enumerate(train_data_loader):
                total_iters += 1

                # add 1 channel
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                x = x.view(-1, 1, patch_size, patch_size)
                y = y.view(-1, 1, patch_size, patch_size)

                pred, out_dist = self.UNAD(x)
                if self.use_dflloss:
                    loss_mse = self.criterion(pred, y)
                    loss_dfl = self.dfl_loss(out_dist, x, y)

                    loss = loss_mse + self.dfl_lossweight * loss_dfl
                    # TODO: print all losses
                else:
                    loss = self.criterion(pred, y)
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
                    # save model at rank 0
                    self.__save_model(total_iters)
                    np.save(os.path.join(self.work_dir, 'loss_{}_iter.npy'.format(total_iters)),
                            np.array(train_losses))

                if total_iters % self.test_interval == 0:
                    self.test()
                    self.UNAD.train()
        self.print('Training finished!')
        self.test()

    def test(self, test_iters=None):
        """Test UNAD."""
        if test_iters:
            self.load_model(test_iters)
        # Only test on rank 0
        if self.rank > 0:
            return
        test_model = self.UNAD
        self.print('Testing with origin model')
        test_model.eval()
        # compute PSNR, SSIM, RMSE
        ori_psnr_avg, ori_ssim_avg, ori_rmse_avg = 0, 0, 0
        pred_psnr_avg, pred_ssim_avg, pred_rmse_avg = 0, 0, 0

        with torch.no_grad():
            for i, (x, y, path) in enumerate(self.test_data_loader):
                shape_ = x.shape[-1]
                x = x.unsqueeze(0).float().to(self.device)
                y = y.unsqueeze(0).float().to(self.device)

                pred, _ = test_model(x)

                # denormalize, truncate
                x = self.__trunc(self.denormalize_(x.view(shape_, shape_).cpu().detach()))
                y = self.__trunc(self.denormalize_(y.view(shape_, shape_).cpu().detach()))
                pred = self.__trunc(self.denormalize_(pred.view(shape_, shape_).cpu().detach()))

                data_range = self.trunc_max - self.trunc_min

                original_result, pred_result = compute_measure(x, y, pred, data_range)
                ori_psnr_avg += original_result[0]
                ori_ssim_avg += original_result[1]
                ori_rmse_avg += original_result[2]
                pred_psnr_avg += pred_result[0]
                pred_ssim_avg += pred_result[1]
                pred_rmse_avg += pred_result[2]

                # save result figure
                if self.result_fig:
                    self.save_fig(x, y, pred, original_result, pred_result, path)

            self.print('\n')
            self.print('Original === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                ori_psnr_avg / len(self.test_data_loader),
                ori_ssim_avg / len(self.test_data_loader),
                ori_rmse_avg / len(self.test_data_loader)))
            self.print('\n')
            self.print('Predictions === \nPSNR avg: {:.4f} \nSSIM avg: {:.4f} \nRMSE avg: {:.4f}'.format(
                pred_psnr_avg / len(self.test_data_loader),
                pred_ssim_avg / len(self.test_data_loader),
                pred_rmse_avg / len(self.test_data_loader)))

    def print(self, msg):
        if self.rank <= 0:
            print(msg)
