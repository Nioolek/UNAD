from typing import List, Tuple, Optional

import os
import re
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class CTDataset(Dataset):
    """Dataset to load slices of CT images.

    Args:
        mode (str): Training mode. The value should be in ['train', 'test','pretrain', 'pretrain_test'].
        load_mode (int): Data loading mode. The value should be 0 or 1.
            If the value is 0, the data will be read from the hard disk when
            it needs to be read. 
            If the value is 1, store all the data in
            memory ahead of time to reduce IO, but this will consume a lot
            of memory space.
        data_root (str): The root path of the data.
        test_patient (str): The name of the test patient.
        patch_n (int): The patch number in current epoch.
        patch_size (int): The patch size in current epoch.
    """

    def __init__(self,
                 mode: str,
                 load_mode: int,
                 data_root: str,
                 test_patient: str,
                 patch_n: Optional[int] = None,
                 patch_size: Optional[int] = None):
        assert mode in ['train', 'test', 'pretrain', 'pretrain_test'], "mode is 'train' , 'test' " \
                                                                       ", 'pretrain' or 'pretrain_test'"
        assert load_mode in [0, 1], "load_mode is 0 or 1"

        input_path = sorted(glob(os.path.join(data_root, '*_input.npy')))
        target_path = sorted(glob(os.path.join(data_root, '*_target.npy')))
        self.load_mode = load_mode
        self.patch_n = patch_n
        self.patch_size = patch_size

        if mode in ['train', 'pretrain']:
            input_ = [f for f in input_path if test_patient not in f]
            target_ = [f for f in target_path if test_patient not in f]

            input_, target_ = self.filter(input_, target_)
            self.input_paths = input_
            if load_mode == 0:  # batch data load
                self.input_ = input_
                self.target_ = target_
            else:  # all data load
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]
        else:  # mode in ['test', 'pretrain_test']
            input_ = [f for f in input_path if test_patient in f]
            target_ = [f for f in target_path if test_patient in f]
            input_, target_ = self.filter(input_, target_)
            self.input_paths = input_
            if load_mode == 0:
                self.input_ = input_
                self.target_ = target_
            else:
                self.input_ = [np.load(f) for f in input_]
                self.target_ = [np.load(f) for f in target_]

    def filter(self, input_: List[str], target_: List[str]):
        """For training, do not need to filter the slices."""
        return input_, target_

    def __len__(self):
        return len(self.target_)

    def __getitem__(self, idx: int):
        """Get the current LDCT (input) and the NDCT (target) slices."""
        input_img, target_img = self.input_[idx], self.target_[idx]

        if self.load_mode == 0:
            input_img, target_img = np.load(input_img), np.load(target_img)

        if self.patch_size:
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size)
            assert input_patches.shape[0] == self.patch_n
            assert len(input_patches.shape) == 3
            input_patches = np.ascontiguousarray(input_patches)
            target_patches = np.ascontiguousarray(target_patches)
            return (input_patches, target_patches)
        else:
            input_path = self.input_paths[idx]
            return (input_img, target_img, input_path)


class CTDatasetPretrain(CTDataset):
    """Dataset to load slices of CT images while pretraining.

    Args:
        load_mode (int): Data loading mode. The value should be 0 or 1.
            If the value is 0, the data will be read from the hard disk when
            it needs to be read. 
            If the value is 1, store all the data in
            memory ahead of time to reduce IO, but this will consume a lot
            of memory space.
        predict_num (Tuple[int, int]): The number of images to be predicted before
            and after the current image. The value is only used while pretraining.
    """

    def __init__(self,
                 *args,
                 load_mode: int = 0,
                 predict_num: Tuple[int, int] = (3, 3),
                 **kwargs):
        self.predict_num = predict_num
        self.pattern = re.compile(r'(.+)/(.+)_(.+)_target.npy')
        # TODO: Support load_mode == 1
        assert load_mode == 0, 'Pretrain only support load_mode == 0'
        super().__init__(*args, load_mode=0, **kwargs)

    def filter(self, input_: List[str], target_: List[str]):
        """For pretraining, the slices which cannot obtain the anterior or posterior
        slices, need to be filtered."""
        res_input_, res_target_ = [], []
        for i, t in zip(input_, target_):
            _, patient, imgind = re.findall(self.pattern, t)[0]
            imgind = int(imgind)
            target_path_first = t.replace('%s_target.npy' % imgind,
                                          '%s_target.npy' % (int(imgind) - self.predict_num[0]))
            target_path_last = t.replace('%s_target.npy' % imgind,
                                         '%s_target.npy' % (int(imgind) + self.predict_num[1]))
            if (not os.path.exists(target_path_first)) or (not os.path.exists(target_path_last)):
                continue

            res_input_.append(i)
            res_target_.append(t)
        return res_input_, res_target_

    def get_img(self, idx: int):
        """Get the current LDCT image and the LDCT images before and after the current image."""
        # Only support load_mode == 0
        input_path, target_path = self.input_[idx], self.target_[idx]
        _, patient, imgind = re.findall(self.pattern, target_path)[0]
        imgind = int(imgind)

        target_path_before = [target_path.replace('%s_target.npy' % imgind, '%s_target.npy' % (int(imgind) - i)) for i
                              in range(self.predict_num[0], 0, -1)]
        target_path_after = [target_path.replace('%s_target.npy' % imgind, '%s_target.npy' % (int(imgind) + i)) for i in
                             range(1, self.predict_num[1] + 1)]
        input_path_before = [i.replace('_target.npy', '_input.npy') for i in target_path_before]
        input_path_after = [i.replace('_target.npy', '_input.npy') for i in target_path_after]
        return np.load(input_path), [np.load(i) for i in input_path_before], [np.load(i) for i in input_path_after]

    def __getitem__(self, idx: int):
        """Get the current LDCT image and the LDCT images before and after the current image.
        Crop the image into patches if patch_size is not None."""
        input_img, input_imgs_before, input_imgs_after = self.get_img(idx)

        if self.patch_size:
            input_patches, patch_input_imgs_before, patch_input_imgs_after = get_patch_pretrain(
                input_img,
                input_imgs_before,
                input_imgs_after,
                self.patch_n,
                self.patch_size)
            assert input_patches.shape[0] == self.patch_n
            assert len(input_patches.shape) == 3
            input_patches = np.ascontiguousarray(input_patches)
            return input_patches, patch_input_imgs_before, patch_input_imgs_after
        else:
            return input_img, np.array(input_imgs_before), np.array(input_imgs_after)


def get_patch(full_input_img: np.ndarray,
              full_target_img: np.ndarray,
              patch_n: int,
              patch_size: int):
    """Randomly get patches from the full image while training."""
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        # Get the coordinates randomly
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # crop img
        patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
        patch_target_img = full_target_img[top:top + new_h, left:left + new_w]
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)
    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_patch_pretrain(full_input_img: np.ndarray,
                       input_imgs_before: List[np.ndarray],
                       input_imgs_after: List[np.ndarray],
                       patch_n: int,
                       patch_size: int):
    """Randomly get patches from the full image while pretraining."""
    assert full_input_img.shape == input_imgs_before[0].shape
    # (patch_n, patch_size, patch_size)
    patch_input_imgs = []
    patch_input_imgs_before = []
    patch_input_imgs_after = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size, patch_size
    for _ in range(patch_n):
        # Get the coordinates randomly
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # crop img
        patch_input_img = full_input_img[top:top + new_h, left:left + new_w]
        patch_input_imgs.append(patch_input_img)
        patch_input_imgs_before.append(np.array([i[top:top + new_h, left:left + new_w] for i in input_imgs_before]))
        patch_input_imgs_after.append(np.array([i[top:top + new_h, left:left + new_w] for i in input_imgs_after]))

    return np.array(patch_input_imgs), np.array(patch_input_imgs_before), np.array(patch_input_imgs_after)


def get_loader(mode: str = 'train', load_mode: int = 0,
               data_root: str = None, test_patient: str = 'L506',
               patch_n: Optional[int] = None, patch_size: Optional[int] = None,
               batch_size: int = 32, num_workers: int = 6,
               predict_num: Tuple[int, int] = (3, 3),
               use_dist: bool = False):
    """Get dataloader for the training, testing, pretraining and pretraining testing.

    Args:
        mode (str): Training mode. The value should be in ['train', 'test',
            'pretrain', 'pretrain_test'].
        load_mode (int): Data loading mode. The value should be 0 or 1.
            If the value is 0, the data will be read from the hard disk when
            it needs to be read. If the value is 1, store all the data in
            memory ahead of time to reduce IO, but this will consume a lot
            of memory space.
        data_root (str): The root path of the data.
        test_patient (str): The name of the test patient.
        patch_n (Optional[int]): The patch number in current epoch.
        patch_size (Optional[int]): The patch size in current epoch.
        batch_size (int): The batch size in current epoch.
        num_workers (int): The number of workers for data loading.
        predict_num (Tuple[int, int]): The number of images to be predicted before
            and after the current image. The value is only used while pretraining.
        use_dist (bool): Whether to use distributed training.
    Returns:
        data_loader (DataLoader): The dataloader.
        data_sampler (Sampler): The sampler for distributed training.
    """
    assert mode in ['train', 'test', 'pretrain', 'pretrain_test']
    if mode in ['test', 'pretrain_test']:
        assert batch_size == 1, 'Now, only support batch_size == 1 while testing.'

    if mode.startswith('pretrain'):
        dataset_ = CTDatasetPretrain(mode=mode, load_mode=load_mode, data_root=data_root,
                                     test_patient=test_patient, patch_n=patch_n, patch_size=patch_size,
                                     predict_num=predict_num)
    else:
        dataset_ = CTDataset(mode=mode, load_mode=load_mode, data_root=data_root,
                             test_patient=test_patient, patch_n=patch_n, patch_size=patch_size)
    shuffle = True if mode in ['train', 'pretrain'] else False
    if use_dist:
        data_sampler = torch.utils.data.distributed.DistributedSampler(dataset_, shuffle=shuffle)
        data_loader = DataLoader(dataset=dataset_, batch_size=batch_size,
                                 num_workers=num_workers,
                                 pin_memory=True, sampler=data_sampler)
    else:
        data_sampler = None
        data_loader = DataLoader(dataset=dataset_, batch_size=batch_size,
                                 shuffle=True if mode in ['train', 'pretrain'] else False,
                                 num_workers=num_workers)
    return data_loader, data_sampler


class LoaderHook(object):
    """Stepwise Patch Increasing Strategy.
    During training, the patch size is gradually increased in stages.

    Args:
        mode (str): Training mode. The value should be in ['train', 'pretrain'].
        load_mode (int): Data loading mode. The value should be 0 or 1.
            If the value is 0, the data will be read from the hard disk when
            it needs to be read. If the value is 1, store all the data in
            memory ahead of time to reduce IO, but this will consume a lot
            of memory space.
        data_root (str): The root path of the data.
        test_patient (str): The name of the test patient.
        num_workers (int): The number of workers for data loading.
        switch_epochs (Tuple): The epochs when the patch size is increased.
        switch_bs (Tuple): The batch size corresponding to each epoch.
        switch_ps (Tuple): The patch size corresponding to each epoch.
        switch_pn (Tuple): The number of patches corresponding to each epoch.
        predict_num (Tuple[int, int]): The number of images to be predicted before
            and after the current image. The value is only used while pretraining.
        use_dist (bool): Whether to use distributed training.
    """

    def __init__(self,
                 data_root: str,
                 mode: str = 'train',
                 load_mode: int = 0,
                 test_patient: str = 'L506',
                 num_workers: int = 6,
                 switch_epochs: Tuple = (0,),
                 switch_bs: Tuple = (8,),
                 switch_ps: Tuple = (64,),
                 switch_pn: Tuple = (5,),
                 predict_num: Tuple[int, int] = (3, 3),
                 use_dist: bool = False):
        self.mode = mode
        assert self.mode in ['train', 'pretrain']
        self.load_mode = load_mode
        self.data_root = data_root
        self.test_patient = test_patient
        self.num_workers = num_workers

        self.switch_epochs = switch_epochs
        self.switch_bs = switch_bs
        self.switch_ps = switch_ps
        self.switch_pn = switch_pn
        self.train_data_loader = None
        self.is_init = False

        self.predict_num = predict_num
        assert len(self.predict_num) == 2

        assert len(switch_epochs) > 0
        assert len(switch_epochs) == len(switch_bs) == len(switch_ps) == len(switch_pn)

        self.patch_size = None
        self.use_dist = use_dist

    def get_data_loader(self, epoch: int):
        """Get the data loader corresponding to the current epoch."""

        if not self.is_init:
            self.train_data_loader, self.train_sampler = get_loader(
                mode=self.mode,
                load_mode=self.load_mode,
                data_root=self.data_root,
                test_patient=self.test_patient,
                num_workers=self.num_workers,
                patch_n=self.switch_pn[0],
                patch_size=self.switch_ps[0],
                batch_size=self.switch_bs[0],
                predict_num=self.predict_num,
                use_dist=self.use_dist)
            self.patch_size = self.switch_ps[0]
            print('Init train dataloader at epoch %s, with batch_size=%s, patch_size=%s, patch_n=%s' % (
                epoch, self.switch_bs[0], self.switch_ps[0], self.switch_pn[0]))
            self.is_init = True
            return self.train_data_loader, self.train_sampler, self.patch_size
        if epoch in self.switch_epochs:
            ind = self.switch_epochs.index(epoch)
            self.train_data_loader, self.train_sampler = get_loader(
                mode=self.mode,
                load_mode=self.load_mode,
                data_root=self.data_root,
                test_patient=self.test_patient,
                num_workers=self.num_workers,
                patch_n=self.switch_pn[ind],
                patch_size=self.switch_ps[ind],
                batch_size=self.switch_bs[ind],
                predict_num=self.predict_num,
                use_dist=self.use_dist)
            print('New data loader at epoch %s, batch_size=%s, patch_size=%s, patch_n=%s' % (
                epoch, self.switch_bs[ind], self.switch_ps[ind], self.switch_pn[ind]))
            self.patch_size = self.switch_ps[ind]
        return self.train_data_loader, self.train_sampler, self.patch_size
