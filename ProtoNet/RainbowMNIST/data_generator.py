import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

class RainbowMNIST(Dataset):

    def __init__(self, args, mode):
        super(RainbowMNIST, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode = mode
        self.data_file = '{}RainbowMNIST/rainbowmnist_all.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.num_groupid = len(self.data.keys())

        for group_id in range(self.num_groupid):
            self.data[group_id]['labels'] = self.data[group_id]['labels'].reshape(10, 100)[:, :20]
            self.data[group_id]['images'] = self.data[group_id]['images'].reshape(10, 100, 28, 28, 3)[:, :20, ...]
            self.data[group_id]['images'] = torch.tensor(np.transpose(self.data[group_id]['images'], (0, 1, 4, 2, 3)))

        if self.mode == 'train':
            self.sel_group_id = np.array([49,  8, 19, 47, 25, 27, 42, 50, 24, 40,  3, 45,  6, 41,  2, 17, 14,
           10,  5, 26, 12, 33,  9, 11, 32, 54, 28,  7, 39, 51, 46, 44, 30, 13,
           18,  0, 34, 43, 52, 29])
            num_of_tasks = self.sel_group_id.shape[0]
            if self.args.ratio<1.0:
                num_of_tasks = int(num_of_tasks*self.args.ratio)
                self.sel_group_id = self.sel_group_id[:num_of_tasks]
        elif self.mode == 'val':
            self.sel_group_id = np.array([15, 16, 38, 36, 37,  4])
        elif self.mode == 'test':
            self.sel_group_id = np.array([35, 48, 23, 20, 22, 55,  1, 21, 31, 53])


    def __len__(self):
        return self.args.metatrain_iterations*self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.arange(self.data[0]['images'].shape[0])
        self.samples_idx = np.arange(self.data[0]['images'].shape[1])

        support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 28, 28)))
        query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 28, 28)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])


        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_group = np.random.choice(self.sel_group_id, size=1, replace=False).item()
            for j in range(10):
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[self.choose_group]['images'][j, choose_samples[:self.k_shot], ...]
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[self.choose_group]['images'][j, choose_samples[
                            self.k_shot:], ...]
                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)