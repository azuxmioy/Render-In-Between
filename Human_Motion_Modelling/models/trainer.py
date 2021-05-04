import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import os
import cv2
import numpy as np
import random

from models.transformer import build_transformer
from models.position_encoding import build_position_encoding
from models.networks import Discriminator_2D, get_scheduler
from models.losses import MaskedMSEloss, MaskedL1loss, GANLoss
from utils.utils import load_state_dict

class MotInterp_Trainer(nn.Module):

    def __init__(self, cfg, resume=False):
        super(MotInterp_Trainer, self).__init__()

        self.cfg = cfg
        self.resume = resume
        self.out_path = cfg.out_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        self._init_models()
        self._init_losses()
        self._init_optimizers()


        print( sum(p.numel() for p in self.transformer.parameters() if p.requires_grad))

    def _init_models(self):
        self.pos_encode = build_position_encoding(self.cfg.pos_encode).to(self.device)
        self.transformer = build_transformer(self.cfg.transformer).to(self.device)
        self.discriminator = Discriminator_2D(self.cfg.discriminator).to(self.device)
        self.start_epoch = -1

        if self.resume:
            load_state_dict(self.transformer, self.cfg.model_pretrain)

            if self.cfg.netD_pretrain != '':
                load_state_dict(self.discriminator, self.cfg.netD_pretrain)

            self.start_epoch = int(self.cfg.model_pretrain[-7:-4]) - 1
            print('Resume from epoch %d' % self.start_epoch)


    def _init_losses(self):
        self.pose2d_criterion = MaskedL1loss(use_mask=True).to(self.device)
        self.criterionGAN_D = GANLoss(use_lsgan=self.cfg.use_lsgan, smooth=self.cfg.gan_smooth).to(self.device)
        self.criterionGAN_G = GANLoss(use_lsgan=self.cfg.use_lsgan, smooth=False).to(self.device)

    def _init_optimizers(self):

        param_groups  = []
        param_groups  += [{'params': self.transformer.parameters(), 'lr_mult': 1.0}]
        #param_groups  += [{'params': self.seq2seq.parameters(), 'lr_mult': 1.0}]


        self.optimizer = torch.optim.Adam(param_groups, lr=self.cfg.lr, 
                                    betas=(self.cfg.beta1, self.cfg.beta2), amsgrad=True)

        self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(),
                                            lr=self.cfg.lr_d, momentum=0.9, weight_decay=1e-4)

        if self.resume and self.cfg.optimizer != '':
            print('restore optimizer...')
            state_dict = torch.load(self.cfg.optimizer)
            self.optimizer.load_state_dict(state_dict['transformer'])
            self.optimizer_D.load_state_dict(state_dict['discriminator'])

        else:
            self.start_epoch = -1


        self.schedulers = []
        self.optimizers = []

        self.optimizers.append(self.optimizer)

        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.cfg, self.start_epoch))
        
    def set_input(self, data):

        #self.input_src = data ['data'].to(self.device)
        #self.input_src = data ['interp'].to(self.device)
        self.input_src = data ['input'].to(self.device)

        self.input_tar = data ['interp'].to(self.device)
        #self.input_tar = data ['interp'].to(self.device)
        #self.input_tar = torch.zeros_like(self.input_src).to(self.device)
        #self.input_tar = torch.randn(self.input_src.shape).to(self.device)
        '''
        not_mask = data['src_mask']
        Bs, C, L = self.input_src.shape
        self.input_tar = data ['data'].to(self.device) *  not_mask.unsqueeze(1).repeat(1, C, 1).float().to(self.device)
        '''
        self.gt = data ['data'].clone().to(self.device).detach()

        self.mask_src = data ['src_mask'].to(self.device)
        #self.mask_src = data ['src_mask'].to(self.device)

        #self.mask_tar = data ['src_mask'].to(self.device)
        self.mask_tar = data ['tar_mask'].to(self.device)
        self.mask_pad = data ['mask'].to(self.device)


    def forward(self):

        self.pos_src = self.pos_encode(self.mask_src).to(self.device)
        self.pos_tar = self.pos_src.clone()

        self.pred, self.reco = self.transformer.forward(
                self.input_src, self.mask_src, self.pos_src, self.input_tar, self.mask_pad, self.pos_tar, self.cfg.train_sample_rate)

        self.pred = self.pred.permute(1, 2, 0)
        self.reco = self.reco.permute(1, 2, 0)


    def backward_D(self):
        pred_real = self.discriminator( self.gt.unsqueeze(1) )
        pred_fake = self.discriminator( self.pred.unsqueeze(1).detach() )

        loss_D_real = self.criterionGAN_D(pred_real, True)
        loss_D_fake = self.criterionGAN_D(pred_fake, False)

        self.loss_D = ( loss_D_real + loss_D_fake ) * self.cfg.gan_w

        self.loss_D.backward()

    def backward(self):
        #self.loss_pose2d = self.pose2d_criterion (self.pred, self.mask_tar, self.gt)
        
        mask_gen = torch.logical_xor(self.mask_src, self.mask_pad).to(self.device)
        mask_gen = ~mask_gen

        self.loss_reco = self.pose2d_criterion (self.reco, self.mask_src, self.gt)
        self.loss_pose2d = self.pose2d_criterion (self.pred, mask_gen, self.gt)
        self.weighted_loss2d = (self.cfg.w_codition * self.loss_reco + self.loss_pose2d ) * self.cfg.w_2d

        pred_fake = self.discriminator ( self.pred.unsqueeze(1) ) 
        self.loss_G_GAN = self.criterionGAN_G(pred_fake, True) * self.cfg.gan_w

        gt_feature = self.discriminator.get_feature(self.gt.unsqueeze(1))
        pred_feature = self.discriminator.get_feature(self.pred.unsqueeze(1))
        self.loss_feat = F.l1_loss(pred_feature, gt_feature) * self.cfg.feat_w


        self.total_loss = self.loss_G_GAN + self.weighted_loss2d + self.loss_feat
        self.total_loss.backward()


    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.optimizer_D.step()


        self.optimizer.zero_grad()
        self.backward()
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 1.0)
        self.optimizer.step()

    def update_learning_rate(self):
        lrs = []
        for scheduler in self.schedulers:
           scheduler.step()
           lrs.append(scheduler.get_last_lr())
        return lrs

    def get_current_losses(self):
        losses = {}


        losses['D/loss_D'] = self.loss_D.data


        losses['recon/denoise'] = self.loss_reco.data
        losses['recon/2D'] = self.loss_pose2d.data
        losses['recon/weighted'] = self.weighted_loss2d.data
        losses['recon/G_GAN'] = self.loss_G_GAN.data
        
        losses['recon/feat'] = self.loss_feat.data
        losses['recon/total'] = self.total_loss.data


        return losses

    def save_network(self, epoch):
        
        save_path = os.path.join(self.out_path, "model_epoch{:03d}.pth".format(epoch))

        torch.save(self.transformer.cpu().state_dict(), save_path)
        
        save_path = os.path.join(self.out_path, "netD_epoch{:03d}.pth".format(epoch))

        torch.save(self.discriminator.cpu().state_dict(), save_path)

        save_path = os.path.join(self.out_path, "opt_epoch{:03d}.pth".format(epoch))

        torch.save({'transformer': self.optimizer.state_dict(), 'discriminator': self.optimizer_D.state_dict()}, save_path)

        self.transformer.to(self.device)
        self.discriminator.to(self.device)


    def inference(self, data, encoder_mask, decoder_mask):

        self.pos_encode.eval()
        self.transformer.eval()

        # data: N * C * L

        data = torch.unsqueeze(data, dim=0).to(self.device)
        target = torch.randn(data.shape).to(self.device)

        encoder_mask = torch.unsqueeze(encoder_mask, dim=0).to(self.device)
        decoder_mask = torch.unsqueeze(decoder_mask, dim=0).to(self.device)

        pos_src = self.pos_encode(encoder_mask).to(self.device)
        pos_tar = self.pos_encode(decoder_mask).to(self.device)

        pred, _ = self.transformer.forward(
                data, encoder_mask, pos_src, target, decoder_mask, pos_tar)

        pred = pred.permute(1, 2, 0)

        return pred

