import os
from torch.utils.data.dataset import Dataset
import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader


class TrainSetDataLoader(Dataset):
    def __init__(self, args):
        super(TrainSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.scale = args.scale_factor
        self.task = args.task
        self.dataset_dir = args.path_for_train + args.task + '_' + str(args.max_angRes) + 'x' + str(args.max_angRes) + '/'
        self.data_list = args.data_list

        self.file_list = []
        for data_name in self.data_list:
            tmp_list = os.listdir(self.dataset_dir + data_name)
            for index, _ in enumerate(tmp_list):
                tmp_list[index] = data_name + '/' + tmp_list[index]

            self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index % self.item_num]]
        with h5py.File(file_name[0], 'r') as hf:
            LF = np.array(hf.get('LF'))  # LF image

        # crop angular resolution
        max_angRes = np.size(LF, 1)
        LF = LF[:, (max_angRes-self.angRes_out)//2:(max_angRes+self.angRes_out)//2,
             (max_angRes-self.angRes_out)//2:(max_angRes+self.angRes_out)//2, :, :]
        LF = torch.from_numpy(LF)
        LF = self.augmentation(LF)

        return LF

    @staticmethod
    def augmentation(label):
        if random.random() < 0.5:  # flip along W-V direction
            label = torch.flip(label, dims=[2, 4])
        if random.random() < 0.5:  # flip along W-V direction
            label = torch.flip(label, dims=[1, 3])
        if random.random() < 0.5:  # transpose between U-V and H-W
            label = label.permute(0, 2, 1, 4, 3)
        return label

    def __len__(self):
        return self.item_num


def MultiTestSetDataLoader(args):
    # get testdataloader of every test dataset
    if args.task == 'SSR':
        data_list = args.data_list
    elif args.task == 'ASR':
        dataset_dir = args.path_for_test + args.task + '_' + str(args.max_angRes) + 'x' + str(args.max_angRes) + '/' + args.data_list[0]
        data_list = os.listdir(dataset_dir)

    test_Loaders = []
    length_of_tests = 0
    for data_name in data_list:
        test_Dataset = TestSetDataLoader(args, data_name)
        length_of_tests += len(test_Dataset)

        test_Loaders.append(DataLoader(dataset=test_Dataset, num_workers=args.num_workers, batch_size=1, shuffle=False))

    return data_list, test_Loaders, length_of_tests


class TestSetDataLoader(Dataset):
    def __init__(self, args, data_name):
        super(TestSetDataLoader, self).__init__()
        self.angRes_in = args.angRes_in
        self.angRes_out = args.angRes_out
        self.task = args.task
        if args.task == 'SSR':
            self.dataset_dir = args.path_for_test + args.task + '_' + str(args.max_angRes) + 'x' + str(args.max_angRes) + '/'
        elif args.task == 'ASR':
            self.dataset_dir = args.path_for_test + args.task + '_' + str(args.max_angRes) + 'x' + str(
                args.max_angRes) + '/' + args.data_list[0] + '/'
        self.data_list = [data_name]

        self.file_list = []
        tmp_list = os.listdir(self.dataset_dir + data_name)
        for index, _ in enumerate(tmp_list):
            tmp_list[index] = data_name + '/' + tmp_list[index]
        self.file_list.extend(tmp_list)

        self.item_num = len(self.file_list)

    def __getitem__(self, index):
        file_name = [self.dataset_dir + self.file_list[index]]
        with h5py.File(file_name[0], 'r') as hf:
            LF = np.array(hf.get('LF'))  # LF image

        # crop angular resolution
        max_angRes = np.size(LF, 1)
        LF = LF[:, (max_angRes-self.angRes_out)//2:(max_angRes+self.angRes_out)//2,
             (max_angRes-self.angRes_out)//2:(max_angRes+self.angRes_out)//2, :, :]
        LF = torch.from_numpy(LF)
        LF_name = self.file_list[index].split('/')[-1].split('.')[0]

        return [LF, LF_name]

    def __len__(self):
        return self.item_num


def flip_SAI(data, angRes):
    if len(data.shape)==2:
        H, W = data.shape
        data = data.reshape(H, W, 1)

    H, W, C = data.shape
    data = data.reshape(angRes, H//angRes, angRes, W//angRes, C) # [U, H, V, W, C]
    data = data[::-1, ::-1, ::-1, ::-1, :]
    data = data.reshape(H, W, C)

    return data


def augmentation(data, label):
    if random.random() < 0.5:  # flip along W-V direction
        data = data[:, ::-1]
        label = label[:, ::-1]
    if random.random() < 0.5:  # flip along W-V direction
        data = data[::-1, :]
        label = label[::-1, :]
    if random.random() < 0.5:  # transpose between U-V and H-W
        data = data.transpose(1, 0)
        label = label.transpose(1, 0)
    return data, label

