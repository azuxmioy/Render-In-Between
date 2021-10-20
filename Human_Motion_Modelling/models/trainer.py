import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import os

from models.transformer import build_transformer
from models.position_encoding import build_position_encoding
from models.discriminator import Discriminator_2D
from models.losses import MaskedL1loss, GANLoss
from utils.utils import load_state_dict

##############################################################################
# Network helper functions
##############################################################################    

def get_scheduler(optimizer, opt, iterations=-1):
    if 'lr_policy' not in opt or opt.lr_policy == 'constant':
        scheduler = None # constant scheduler

    elif opt.lr_policy == 'lambda': # learning rate warmup setting
        def lambda_rule(epoch):
            lr_l = min ( (epoch+1) ** -0.5, (epoch+1) * opt.warmup ** -1.5)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.step_size,
                                 gamma=opt.gamma, last_epoch=iterations)
    elif opt.lr_policy == 'multistep':
        step = opt.step_size
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[step, step+step//2, step+step//2+step//4],
                                        gamma=opt.gamma, last_epoch=iterations)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

##############################################################################
# Trainer Class
############################################################################# 

class MotInterp_Trainer(nn.Module):

    def __init__(self, cfg, resume=False):
        super(MotInterp_Trainer, self).__init__()

        self.cfg = cfg
        self.resume = resume
        self.out_path = cfg.out_dir # save network path
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # Experimental motion discriminator
        self.use_dis = cfg.use_dis

        self._init_models()
        self._init_losses()
        self._init_optimizers()

        print( sum(p.numel() for p in self.transformer.parameters() if p.requires_grad))

    def _init_models(self):
        self.pos_encode = build_position_encoding(self.cfg.pos_encode).to(self.device)
        self.transformer = build_transformer(self.cfg.transformer).to(self.device)
        
        if self.use_dis:
            self.discriminator = Discriminator_2D(self.cfg.discriminator).to(self.device)

        
        self.start_epoch = -1

        if self.resume:
            load_state_dict(self.transformer, self.cfg.model_pretrain)
            if self.use_dis and self.cfg.netD_pretrain != '':
                load_state_dict(self.discriminator, self.cfg.netD_pretrain)

            self.start_epoch = int(self.cfg.model_pretrain[-7:-4]) - 1
            print('Resume from epoch %d' % self.start_epoch)


    def _init_losses(self):
        self.pose2d_criterion = MaskedL1loss(use_mask=True).to(self.device)
        if self.use_dis:
            self.criterionGAN_D = GANLoss(use_lsgan=self.cfg.use_lsgan, smooth=self.cfg.gan_smooth).to(self.device)
            self.criterionGAN_G = GANLoss(use_lsgan=self.cfg.use_lsgan, smooth=False).to(self.device)


    def _init_optimizers(self):

        param_groups  = []
        param_groups  += [{'params': self.transformer.parameters(), 'lr_mult': 1.0}]

        self.optimizer = torch.optim.Adam(param_groups, lr=self.cfg.lr, 
                                    betas=(self.cfg.beta1, self.cfg.beta2), amsgrad=True)
        if self.use_dis:
            self.optimizer_D = torch.optim.SGD(self.discriminator.parameters(),
                                            lr=self.cfg.lr*4, momentum=0.9, weight_decay=1e-4)

        if self.resume and self.cfg.optimizer != '':
            print('restore optimizer...')
            state_dict = torch.load(self.cfg.optimizer)
            self.optimizer.load_state_dict(state_dict['transformer'])
            if self.use_dis:
                self.optimizer_D.load_state_dict(state_dict['discriminator'])
        else:
            self.start_epoch = -1


        self.schedulers = []
        self.optimizers = []

        self.optimizers.append(self.optimizer)
        if self.use_dis:
            self.optimizers.append(self.optimizer_D)

        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.cfg, self.start_epoch))

    def set_input(self, data):

        self.input_src = data ['input'].to(self.device)

        self.input_tar = data ['interp'].to(self.device)

        self.gt = data ['data'].clone().to(self.device).detach()

        self.mask_src = data ['src_mask'].to(self.device)
        self.mask_tar = data ['tar_mask'].to(self.device)

        self.mask_pad = data ['mask'].to(self.device)

    def forward(self):

        self.pos_src = self.pos_encode(self.mask_src).to(self.device)
        self.pos_tar = self.pos_src.clone()

        self.pred, self.reco = self.transformer.forward(
                self.input_src, self.mask_src, self.pos_src, self.input_tar, self.mask_pad, self.pos_tar, self.cfg.train_sample_rate)

        self.pred = self.pred.permute(1, 2, 0)
        self.reco = self.reco.permute(1, 2, 0)


    def backward(self):
        #self.loss_pose2d = self.pose2d_criterion (self.pred, self.mask_tar, self.gt)
        
        mask_gen = torch.logical_xor(self.mask_src, self.mask_pad).to(self.device)
        mask_gen = ~mask_gen
        self.loss_reco = self.pose2d_criterion (self.reco, self.mask_src, self.gt)
        self.loss_pose2d = self.pose2d_criterion (self.pred, mask_gen, self.gt)
        self.weighted_loss2d = (self.cfg.w_codition * self.loss_reco + self.loss_pose2d ) * self.cfg.w_2d

        if self.use_dis:
            pred_fake = self.discriminator ( self.pred.unsqueeze(1) ) 
            self.loss_G_GAN = self.criterionGAN_G(pred_fake, True) * self.cfg.w_gan

        self.total_loss = self.weighted_loss2d+self.loss_G_GAN if self.use_dis else self.weighted_loss2d 

        self.total_loss.backward()

    def backward_D(self):
        pred_real = self.discriminator( self.gt.unsqueeze(1) )
        pred_fake = self.discriminator( self.pred.unsqueeze(1).detach() )

        loss_D_real = self.criterionGAN_D(pred_real, True)
        loss_D_fake = self.criterionGAN_D(pred_fake, False)

        self.loss_D = ( loss_D_real + loss_D_fake ) * self.cfg.w_gan

        self.loss_D.backward()

    def optimize_parameters(self):
        self.forward()

        if self.use_dis:
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

        losses['recon/denoise'] = self.loss_reco.data
        losses['recon/2D'] = self.loss_pose2d.data
        losses['recon/weighted'] = self.weighted_loss2d.data
        losses['recon/total'] = self.total_loss.data

        if self.use_dis:
            losses['GAN/loss_D'] = self.loss_D.data
            losses['GAN/loss_G'] = self.loss_G_GAN.data

        return losses

    def save_network(self, epoch):
        
        save_path = os.path.join(self.out_path, "model_epoch{:03d}.pth".format(epoch))
        torch.save(self.transformer.cpu().state_dict(), save_path)
    
        self.transformer.to(self.device)

        if self.use_dis:
            save_path = os.path.join(self.out_path, "netD_epoch{:03d}.pth".format(epoch))
            torch.save(self.discriminator.cpu().state_dict(), save_path)

            self.discriminator.to(self.device)

            save_path = os.path.join(self.out_path, "opt_epoch{:03d}.pth".format(epoch))
            torch.save({'transformer': self.optimizer.state_dict(), 'discriminator': self.optimizer_D.state_dict()}, save_path)
        else:
            save_path = os.path.join(self.out_path, "opt_epoch{:03d}.pth".format(epoch))
            torch.save({'transformer': self.optimizer.state_dict()}, save_path)