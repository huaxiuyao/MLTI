import random
import numpy as np
import torch
from torch.utils.data import Dataset
import pickle

identity = lambda x: x

class Metabolism(Dataset):

    def __init__(self, args, mode):
        super(Metabolism, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode = mode
        self.data_file = '{}/metabolism_data_new.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.num_groupid = len(self.data.keys())

        self.index_label = {}

        self.all_group_list = ['CYP1A2_Veith', 'CYP3A4_Veith', 'CYP2D6_Veith', 'CYP2C9_Substrate_CarbonMangels',
                               'CYP2D6_Substrate_CarbonMangels', 'CYP3A4_Substrate_CarbonMangels', 'CYP2C19_Veith',
                               'CYP2C9_Veith']

        if self.mode == 'train':
            self.sel_group_id = np.array(['CYP1A2_Veith', 'CYP3A4_Veith', 'CYP2D6_Veith', 'CYP2C9_Substrate_CarbonMangels',
                               'CYP2D6_Substrate_CarbonMangels'])
        elif self.mode == 'val':
            self.sel_group_id = np.array(['CYP3A4_Substrate_CarbonMangels', 'CYP2C19_Veith', 'CYP2C9_Veith'])
        elif self.mode == 'test':
            self.sel_group_id = np.array(['CYP3A4_Substrate_CarbonMangels', 'CYP2C19_Veith', 'CYP2C9_Veith'])

        for group_id in self.all_group_list:
            self.data[group_id]['label'] = self.data[group_id]['label']
            self.data[group_id]['feature'] = self.data[group_id]['feature']

            self.index_label[group_id] = {}
            if 'Substrate' not in group_id:
                self.index_label[group_id][0] = np.nonzero(self.data[group_id]['label'] == 0.0)[0]
                self.index_label[group_id][1] = np.nonzero(self.data[group_id]['label'] == 1.0)[0]
            else:
                self.index_label[group_id][0] = np.nonzero(self.data[group_id]['label'] == 0.0)[0]
                self.index_label[group_id][1] = np.nonzero(self.data[group_id]['label'] == 1.0)[0]


    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = 2
        support_x = np.zeros((self.args.meta_batch_size, self.set_size, 1024))
        query_x = np.zeros((self.args.meta_batch_size, self.query_size, 1024))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_group = np.random.choice(self.sel_group_id, size=1, replace=False).item()
            for j in range(2):
                self.samples_idx = np.array(self.index_label[self.choose_group][j])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = \
                    self.data[self.choose_group]['feature'][choose_samples[:self.k_shot], ...].astype(float)
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = \
                    self.data[self.choose_group]['feature'][choose_samples[self.k_shot:], ...].astype(float)

                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        return torch.FloatTensor(support_x), torch.LongTensor(support_y), torch.FloatTensor(query_x), torch.LongTensor(
            query_y)




class NCI(Dataset):

    def __init__(self, args, mode):
        super(NCI, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode = mode
        self.data_file = '{}/NCI_data_new.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.num_groupid = len(self.data.keys())

        self.index_label = {}

        self.all_group_list = [81, 41, 83, 47, 109, 145, 33, 1, 123]

        if self.mode == 'train':
            self.sel_group_id = np.array([81, 41, 83, 47, 109, 145])
            assert self.args.ratio == 1.0
        elif self.mode == 'val':
            self.sel_group_id = np.array([33, 1, 123])
        elif self.mode == 'test':
            self.sel_group_id = np.array([33, 1, 123])

        for group_id in self.all_group_list :
            self.data[group_id]['label'] = self.data[group_id]['label']
            self.data[group_id]['feature'] = self.data[group_id]['feature']

            self.index_label[group_id] = {}
            if group_id in [81, 41, 83, 47, 109, 145]:
                self.index_label[group_id][0] = np.nonzero(self.data[group_id]['label'] == -1.0)[0][:500]
                self.index_label[group_id][1] = np.nonzero(self.data[group_id]['label'] == 1.0)[0][:500]
            else:
                self.index_label[group_id][0] = np.nonzero(self.data[group_id]['label'] == -1.0)[0]
                self.index_label[group_id][1] = np.nonzero(self.data[group_id]['label'] == 1.0)[0]


    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = 2
        support_x = np.zeros((self.args.meta_batch_size, self.set_size, 1024))
        query_x = np.zeros((self.args.meta_batch_size, self.query_size, 1024))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_group = np.random.choice(self.sel_group_id, size=1, replace=False).item()
            for j in range(2):
                self.samples_idx = np.array(self.index_label[self.choose_group][j])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = \
                    self.data[self.choose_group]['feature'][choose_samples[:self.k_shot], ...].astype(float)
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = \
                    self.data[self.choose_group]['feature'][choose_samples[self.k_shot:], ...].astype(float)

                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        return torch.FloatTensor(support_x), torch.LongTensor(support_y), torch.FloatTensor(query_x), torch.LongTensor(query_y)
