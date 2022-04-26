import os
import numpy as np
import torch
from torch.utils.data import Dataset
from preprocess import MacaData
from map_GO import get_go2gene

identity = lambda x: x


def create_go_mask(adata, go2gene):
    genes = adata.var_names
    gene2index = {g: i for i, g in enumerate(genes)}
    GO_IDs = sorted(go2gene.keys())
    go_mask = []
    for go in GO_IDs:
        go_genes = go2gene[go]
        go_mask.append([gene2index[gene] for gene in go_genes])
    return go_mask

class TM(Dataset):

    def __init__(self, args, mode, min_samples=20):
        super(TM, self).__init__()
        self.args = args
        self.nb_classes = args.num_classes
        self.nb_samples_per_class = args.update_batch_size + args.update_batch_size_eval
        self.n_way = args.num_classes
        self.k_shot = args.update_batch_size
        self.k_query = args.update_batch_size_eval
        self.set_size = self.n_way * self.k_shot
        self.query_size = self.n_way * self.k_query
        self.mode = mode
        self.samples_all, self.targets_all, self.go_masks = self.load_tabular_muris(root=args.datadir, mode=mode, min_samples=min_samples)
        self.classes_idx = np.unique(self.targets_all)
        print(len(self.classes_idx))
        self.data = {cl: self.samples_all[self.targets_all == cl, ...] for cl in self.classes_idx}

    def load_tabular_muris(self, root='./filelists/tabula_muris', mode='train', min_samples=20):
        train_tissues = ['BAT', 'MAT', 'Limb_Muscle', 'Trachea', 'Heart', 'Spleen', 'GAT', 'SCAT', 'Mammary_Gland',
                         'Liver', 'Kidney', 'Bladder', 'Brain_Myeloid', 'Brain_Non-Myeloid', 'Diaphragm']

        val_tissues = ["Skin", "Lung", "Thymus", "Aorta"]
        test_tissues = ["Large_Intestine", "Marrow", "Pancreas", "Tongue"]
        split = {'train': train_tissues,
                 'val': val_tissues,
                 'test': test_tissues}
        adata = MacaData(src_file=os.path.join(root, "tabula-muris-comet.h5ad")).adata
        tissues = split[mode]
        # subset data based on target tissues
        adata = adata[adata.obs['tissue'].isin(tissues)]

        filtered_index = adata.obs.groupby(["label"]) \
            .filter(lambda group: len(group) >= min_samples) \
            .reset_index()['index']
        adata = adata[filtered_index]

        # convert gene to torch tensor x
        samples = adata.to_df().to_numpy(dtype=np.float32)
        # convert label to torch tensor y
        targets = adata.obs['label'].cat.codes.to_numpy(dtype=np.int32)
        # 153 go, each go have some genes
        go2gene = get_go2gene(adata=adata, GO_min_genes=64, GO_max_genes=None, GO_min_level=3, GO_max_level=3)

        go_mask = create_go_mask(adata, go2gene)
        # samples: (65812, 2866), 2866 genes, targets: (25065, 1), all 57 labels for training, 153 go
        return samples, targets, go_mask

    def __len__(self):
        return self.args.metatrain_iterations * self.args.meta_batch_size

    def __getitem__(self, index):
        support_x = np.zeros((self.args.meta_batch_size, self.set_size, 2866))
        query_x = np.zeros((self.args.meta_batch_size, self.query_size, 2866))

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

        return torch.FloatTensor(support_x), torch.LongTensor(support_y), torch.FloatTensor(query_x), torch.LongTensor(query_y)