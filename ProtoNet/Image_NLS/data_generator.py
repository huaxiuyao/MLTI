import csv
import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms
import pickle
import ipdb

class DermNet(Dataset):

    def __init__(self, args, mode):
        super(DermNet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes
        self.k_shot = args.update_batch_size
        self.k_query = args.update_batch_size_eval
        self.set_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        self.data_file = '{}/DermNet/Dermnet_all_84.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))
        num_data = [(eachkey, self.data[eachkey].shape[0]) for eachkey in self.data]

        num_data = sorted(num_data, key = lambda x: x[1], reverse=True)

        if mode == 'train':
            sel_class_num = int(self.args.ratio*150)
            print(sel_class_num)
            self.used_diseases = [eachid[0] for eachid in num_data[:sel_class_num]]
        elif mode == 'test':
            self.used_diseases = [eachid[0] for eachid in num_data[150:]]

        for eachkey in self.data.keys():
            self.data[eachkey] = torch.tensor(np.transpose(self.data[eachkey] / np.float32(255), (0, 3, 1, 2)))

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.array(self.used_diseases)

        if self.args.train:
            support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
            query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))

            support_y = np.zeros([self.args.meta_batch_size, self.set_size])
            query_y = np.zeros([self.args.meta_batch_size, self.query_size])

            for meta_batch_id in range(self.args.meta_batch_size):
                self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
                for j in range(self.nb_classes):
                    self.samples_idx = np.arange(self.data[self.choose_classes[j]].shape[0])
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                self.k_shot:], ...]
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        else:
            support_x = torch.FloatTensor(torch.zeros((self.set_size, 3, 84, 84)))
            support_y = np.zeros([self.set_size])
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            query_size_test = sum([self.data[self.choose_classes[j]].shape[0] for j in range(self.nb_classes)]) - self.set_size
            query_x = torch.FloatTensor(torch.zeros((query_size_test, 3, 84, 84)))
            query_y = np.zeros([query_size_test])

            split_loc_pre = [self.data[self.choose_classes[j]].shape[0]-self.k_shot for j in range(self.nb_classes)]

            query_split_loc_list = [sum(split_loc_pre[:j]) for j in range(self.nb_classes+1)]

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(self.data[self.choose_classes[j]].shape[0])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx
                # idx1 = idx[0:self.k_shot + self.k_query]
                support_x[j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                    self.choose_classes[
                        j]][choose_samples[
                            :self.k_shot], ...]
                support_y[j * self.k_shot:(j + 1) * self.k_shot] = j

                query_x[query_split_loc_list[j]:query_split_loc_list[j+1]] = self.data[self.choose_classes[j]][
                    choose_samples[self.k_shot:], ...]
                query_y[query_split_loc_list[j]:query_split_loc_list[j+1]] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)


class ISIC(Dataset):

    def __init__(self, args, mode):
        super(ISIC, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes
        self.k_shot = args.update_batch_size
        self.k_query = args.update_batch_size_eval
        self.set_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        if mode == 'train':
            self.data_file = '{}/ISIC/ISIC_train.pkl'.format(args.datadir)
        elif mode == 'test':
            self.data_file = '{}/ISIC/ISIC_test.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        for eachkey in self.data.keys():
            self.data[eachkey] = torch.tensor(np.transpose(self.data[eachkey] / np.float32(255), (0, 3, 1, 2)))

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        self.classes_idx = np.array(list(self.data.keys()))

        if self.args.train:
            support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
            query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))

            support_y = np.zeros([self.args.meta_batch_size, self.set_size])
            query_y = np.zeros([self.args.meta_batch_size, self.query_size])

            for meta_batch_id in range(self.args.meta_batch_size):
                self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
                for j in range(self.nb_classes):
                    self.samples_idx = np.arange(self.data[self.choose_classes[j]].shape[0])
                    np.random.shuffle(self.samples_idx)
                    choose_samples = self.samples_idx[:self.nb_samples_per_class]
                    # idx1 = idx[0:self.k_shot + self.k_query]
                    support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                :self.k_shot], ...]
                    query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                        self.choose_classes[
                            j]][choose_samples[
                                self.k_shot:], ...]
                    support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                    query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        else:
            support_x = torch.FloatTensor(torch.zeros((self.set_size, 3, 84, 84)))
            support_y = np.zeros([self.set_size])
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            query_size_test = self.data[self.choose_classes[0]].shape[0] + self.data[self.choose_classes[1]].shape[
                0] - self.set_size
            # print(query_size_test, self.data[self.choose_classes[0]].shape[0], self.data[self.choose_classes[1]].shape[0])
            query_x = torch.FloatTensor(torch.zeros((query_size_test, 3, 84, 84)))
            query_y = np.zeros([query_size_test])

            for j in range(self.nb_classes):
                self.samples_idx = np.arange(self.data[self.choose_classes[j]].shape[0])
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx
                # idx1 = idx[0:self.k_shot + self.k_query]
                support_x[j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                    self.choose_classes[
                        j]][choose_samples[
                            :self.k_shot], ...]
                support_y[j * self.k_shot:(j + 1) * self.k_shot] = j

                query_split_loc = self.data[self.choose_classes[0]].shape[0]-self.k_shot

                if j==0:
                    query_x[:query_split_loc] = self.data[self.choose_classes[0]][
                        choose_samples[self.k_shot:], ...]
                    query_y[:query_split_loc] = j
                else:
                    query_x[query_split_loc:] = self.data[self.choose_classes[1]][
                        choose_samples[self.k_shot:], ...]
                    query_y[query_split_loc:] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

class MiniImagenet(Dataset):

    def __init__(self, args, mode):
        super(MiniImagenet, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes  # n-way
        self.k_shot = args.update_batch_size  # k-shot
        self.k_query = args.update_batch_size_eval  # for evaluation
        self.set_size = self.n_way * self.k_shot  # num of samples per set
        self.query_size = self.n_way * self.k_query  # number of samples per set for evaluation
        self.mode = mode
        if mode == 'train':
            self.data_file = '{}/miniImagenet/mini_imagenet_train.pkl'.format(args.datadir)
        elif mode == 'val':
            self.data_file = '{}/miniImagenet/mini_imagenet_val.pkl'.format(args.datadir)
        elif mode == 'test':
            self.data_file = '{}/miniImagenet/mini_imagenet_test.pkl'.format(args.datadir)

        self.data = pickle.load(open(self.data_file, 'rb'))

        self.all_train_classes = np.array([26, 59, 14, 16, 17, 52,  8, 39, 46, 32, 20, 57, 34, 25, 63, 31, 30,
       40,  0, 43,  7, 33, 12,  6, 22, 23, 49, 50, 15, 13, 51, 10, 24, 27,
       47, 55,  9,  5, 18, 36, 44, 35,  4, 21, 61, 42, 11,  3, 45, 58, 60,
       56,  1, 28, 48, 54, 37, 19, 62, 41, 38,  2, 53, 29])
        self.num_train_use_class = int(64*self.args.ratio)

        self.data = torch.tensor(np.transpose(self.data / np.float32(255), (0, 1, 4, 2, 3)))



    def __len__(self):
        return self.args.metatrain_iterations*self.args.meta_batch_size

    def __getitem__(self, index):
        if self.mode == 'train':
            self.classes_idx = self.all_train_classes[:self.num_train_use_class]
        else:
            self.classes_idx = np.arange(self.data.shape[0])

        # ipdb.set_trace()
        self.samples_idx = np.arange(self.data.shape[1])

        support_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.set_size, 3, 84, 84)))
        query_x = torch.FloatTensor(torch.zeros((self.args.meta_batch_size, self.query_size, 3, 84, 84)))

        support_y = np.zeros([self.args.meta_batch_size, self.set_size])
        query_y = np.zeros([self.args.meta_batch_size, self.query_size])

        for meta_batch_id in range(self.args.meta_batch_size):
            self.choose_classes = np.random.choice(self.classes_idx, size=self.nb_classes, replace=False)
            for j in range(self.nb_classes):
                np.random.shuffle(self.samples_idx)
                choose_samples = self.samples_idx[:self.nb_samples_per_class]
                support_x[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = self.data[
                    self.choose_classes[
                        j], choose_samples[
                            :self.k_shot], ...]
                query_x[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = self.data[
                    self.choose_classes[
                        j], choose_samples[
                            self.k_shot:], ...]
                support_y[meta_batch_id][j * self.k_shot:(j + 1) * self.k_shot] = j
                query_y[meta_batch_id][j * self.k_query:(j + 1) * self.k_query] = j

        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)