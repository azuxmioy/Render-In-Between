import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import Variable

def kl_loss(code):
    return torch.mean(torch.pow(code, 2))


def pairwise_cosine_similarity(seqs_i, seqs_j):
    # seqs_i, seqs_j: [batch, statics, channel]
    n_statics = seqs_i.size(1)
    seqs_i_exp = seqs_i.unsqueeze(2).repeat(1, 1, n_statics, 1)
    seqs_j_exp = seqs_j.unsqueeze(1).repeat(1, n_statics, 1, 1)
    return F.cosine_similarity(seqs_i_exp, seqs_j_exp, dim=3)


def temporal_pairwise_cosine_similarity(seqs_i, seqs_j):
    # seqs_i, seqs_j: [batch, channel, time]
    seq_len = seqs_i.size(2)
    seqs_i_exp = seqs_i.unsqueeze(3).repeat(1, 1, 1, seq_len)
    seqs_j_exp = seqs_j.unsqueeze(2).repeat(1, 1, seq_len, 1)
    return F.cosine_similarity(seqs_i_exp, seqs_j_exp, dim=1)


def consecutive_cosine_similarity(seqs):
    # seqs: [batch, channel, time]
    seqs_roll = seqs.roll(shifts=1, dim=2)[1:]
    seqs = seqs[:-1]
    return F.cosine_similarity(seqs, seqs_roll)


def triplet_margin_loss(seqs_a, seqs_b, neg_range=(0.0, 0.5), margin=0.2):
    # seqs_a, seqs_b: [batch, channel, time]

    neg_start, neg_end = neg_range
    batch_size, _, seq_len = seqs_a.size()
    n_neg_all = seq_len ** 2
    n_neg = int(round(neg_end * n_neg_all))
    n_neg_discard = int(round(neg_start * n_neg_all))

    batch_size, _, seq_len = seqs_a.size()
    sim_aa = temporal_pairwise_cosine_similarity(seqs_a, seqs_a)
    sim_bb = temporal_pairwise_cosine_similarity(seqs_b, seqs_a)
    sim_ab = temporal_pairwise_cosine_similarity(seqs_a, seqs_b)
    sim_ba = sim_ab.transpose(1, 2)

    diff_ab = (sim_ab - sim_aa).reshape(batch_size, -1)
    diff_ba = (sim_ba - sim_bb).reshape(batch_size, -1)
    diff = torch.cat([diff_ab, diff_ba], dim=0)
    diff, _ = diff.topk(n_neg, dim=-1, sorted=True)
    diff = diff[:, n_neg_discard:]

    loss = diff + margin
    loss = loss.clamp(min=0.)
    loss = loss.mean()

    return loss

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, smooth=False, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.smooth = smooth
        self.Tensor = tensor

        if use_lsgan:
            self.criterion = nn.MSELoss().cuda()
        else:
            self.criterion = nn.BCELoss().cuda()

    def get_target_tensor(self, input, target_is_real):
        real_label = 1.0
        fake_label = 0.0
        if self.smooth:
            real_label = random.uniform(0.9,1.0)
            fake_label = random.uniform(0.0,0.1)

        if target_is_real:
            real_tensor = self.Tensor(input.size()).fill_(real_label)
            target_tensor = Variable(real_tensor, requires_grad=False).cuda()
        else:
            fake_tensor = self.Tensor(input.size()).fill_(fake_label)
            target_tensor = Variable(fake_tensor, requires_grad=False).cuda()
        return target_tensor

    def __call__(self, input, target_is_real):
        # single prediction:
        if isinstance(input, torch.Tensor):
            target_tensor = self.get_target_tensor(input, target_is_real)
            return self.criterion(input, target_tensor)
        elif isinstance (input, list):
            loss = 0
            for i, pred in enumerate(input):
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.criterion(pred, target_tensor)
            return loss
        else:
            raise NotImplementedError("[Error] unsupport gan input")


class MaskedMSEloss(nn.Module):

    def __init__(self, use_mask = True):
        super(MaskedMSEloss, self).__init__()
        self.use_mask = use_mask

    def forward(self, inputs, mask, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            mask : human mask for higher reconstruction loss
            targets: ground truth labels with shape (num_classes)
        """
        if not self.use_mask:
            return F.mse_loss(inputs, targets)

        Bs, C, L = inputs.shape
        
        not_mask = ~mask
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

        N = not_mask.sum(dtype=torch.float32)

        loss = F.mse_loss(inputs * not_mask, targets * not_mask, reduction='sum') / N

        return loss

class MaskedL1loss(nn.Module):

    def __init__(self, use_mask = True):
        super(MaskedL1loss, self).__init__()
        self.use_mask = use_mask

    def forward(self, inputs, mask, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            mask : human mask for higher reconstruction loss
            targets: ground truth labels with shape (num_classes)
        """
        if not self.use_mask:
            return F.l1_loss(inputs, targets)

        Bs, C, L = inputs.shape
        
        not_mask = ~mask
        not_mask = not_mask.unsqueeze(1).repeat(1, C, 1).float()

        N = not_mask.sum(dtype=torch.float32)

        loss = F.l1_loss(inputs * not_mask, targets * not_mask, reduction='sum') / N

        return loss