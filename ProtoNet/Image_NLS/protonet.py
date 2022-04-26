import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import euclidean_dist
import numpy as np
from torch.distributions import Beta


class Protonet(nn.Module):
    def __init__(self, args, learner):
        super(Protonet, self).__init__()
        self.args = args
        self.learner = learner
        self.dist = Beta(torch.FloatTensor([2]), torch.FloatTensor([2]))

    def forward(self, xs, ys, xq, yq):
        x = torch.cat([xs, xq], 0)

        z = self.learner(x)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(1)
        zq = z[self.args.num_classes * self.args.update_batch_size:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = []
        for i in range(self.args.num_classes * self.args.update_batch_size_eval):
            loss_val.append(-log_p_y[i, yq[i]])

        loss_val = torch.stack(loss_val).squeeze().mean()

        _, y_hat = log_p_y.max(1)

        acc_val = torch.eq(y_hat, yq).float().mean()

        return loss_val, acc_val

    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam.cpu())
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        # uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def mixup_data(self, xs, xq, lam):
        mixed_x = xq.clone()
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(xq.size(), lam)

        mixed_x[:, :, bbx1:bbx2, bby1:bby2] = xs[:, :, bbx1:bbx2, bby1:bby2]

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (xq.size()[-1] * xq.size()[-2]))

        return mixed_x, lam

    def forward_crossmix(self, x1s, y1s, x1q, y1q, x2s, y2s, x2q, y2q):
        lam_mix = self.dist.sample().to("cuda")

        task_2_shuffle_id = np.arange(self.args.num_classes)
        np.random.shuffle(task_2_shuffle_id)
        task_2_shuffle_id_s = np.array(
            [np.arange(self.args.update_batch_size) + task_2_shuffle_id[idx] * self.args.update_batch_size for idx in
             range(self.args.num_classes)]).flatten()
        task_2_shuffle_id_q = np.array(
            [np.arange(self.args.update_batch_size_eval) + task_2_shuffle_id[idx] * self.args.update_batch_size_eval for
             idx in range(self.args.num_classes)]).flatten()

        x2s = x2s[task_2_shuffle_id_s]
        x2q = x2q[task_2_shuffle_id_q]

        x_mix_s, _ = self.mixup_data(x1s, x2s, lam_mix)

        x_mix_q, _ = self.mixup_data(x1q, x2q, lam_mix)

        x = torch.cat([x_mix_s, x_mix_q], 0)

        z = self.learner(x)

        z_dim = z.size(-1)

        z_proto = z[:self.args.num_classes * self.args.update_batch_size].view(self.args.num_classes,
                                                                               self.args.update_batch_size, z_dim).mean(
            1)
        zq = z[self.args.num_classes * self.args.update_batch_size:]

        dists = euclidean_dist(zq, z_proto)

        log_p_y = F.log_softmax(-dists, dim=1)

        loss_val = []
        for i in range(self.args.num_classes * self.args.update_batch_size_eval):
            loss_val.append(-log_p_y[i, y1q[i]])

        loss_val = torch.stack(loss_val).squeeze().mean()

        _, y_hat = log_p_y.max(1)

        acc_val = torch.eq(y_hat, y1q).float().mean()

        return loss_val, acc_val