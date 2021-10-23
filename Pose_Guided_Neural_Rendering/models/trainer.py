import torch
import torch.nn as nn
import os
from torch.optim import lr_scheduler

from models.generator import Generator
from models.discriminator import Discriminator

from models.losses import FeatureMatchingLoss, PerceptualLoss, GANLoss, MaskRegulationLoss, MaskedL1loss
from utils.utils import load_state_dict, tensor2images

##############################################################################
# Network helper functions
############################################################################## 

def get_scheduler(optimizer, opt, iterations=-1):
    if 'lr_policy' not in opt or opt.lr_policy == 'constant':
        scheduler = None # constant scheduler

    elif opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1- opt.step_size) / float(opt.max_epoch-opt.step_size)
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
# Main trainer class
############################################################################## 

class Motion_recovery_auto(nn.Module):

    def __init__(self, cfg, resume=False):
        super(Motion_recovery_auto, self).__init__()

        self.cfg = cfg
        self.resume = resume
        self.out_path = cfg.out_dir
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.add_dis_cfg = getattr(self.cfg.dis, 'additional_discriminators', None)
        self._init_models()
        self._init_losses()
        self._init_optimizers()

        #print( sum(p.numel() for p in self.net_G.parameters() if p.requires_grad))
        #print( sum(p.numel() for p in self.net_D.parameters() if p.requires_grad))

    def _init_models(self):
        self.net_G = Generator(self.cfg.gen).to(self.device)
        self.net_D = Discriminator(self.cfg.dis).to(self.device)

        self.start_epoch = -1

        if self.resume:
            load_state_dict(self.net_G , self.cfg.model_pretrain_G)
            load_state_dict(self.net_D , self.cfg.model_pretrain_D)

            self.start_epoch = int(self.cfg.model_pretrain_G[-7:-4]) - 1
            print('Resume from epoch %d' % self.start_epoch)

    def _init_losses(self):
        self.criterionGAN= GANLoss(self.cfg.gan_mode).to(self.device)
        self.criterion_matching = FeatureMatchingLoss().to(self.device)

        perceptual_cfg = self.cfg.perceptual
        self.criterion_perceptual = PerceptualLoss(
                                  network=perceptual_cfg.model,
                                  layers=perceptual_cfg.layers,
                                  weights=perceptual_cfg.weights,
                                  criterion=perceptual_cfg.criterion,
                                  num_scales=perceptual_cfg.num_scales).to(self.device)

        self.criterion_L1 = nn.L1Loss().to(self.device)
        self.criterion_L1_masked = MaskedL1loss(use_mask = True).to(self.device)
        self.mask_regulation = MaskRegulationLoss()

    def _init_optimizers(self):


        self.optimizer_G = torch.optim.Adam(self.net_G.parameters(), lr=self.cfg.lr, 
                                    betas=(self.cfg.beta1, self.cfg.beta2), amsgrad=True)

        self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=self.cfg.lr_d,
                                    betas=(self.cfg.beta1, self.cfg.beta2), amsgrad=True)

        if self.resume and self.cfg.optimizer != '':
            print('restore optimizer...')
            state_dict = torch.load(self.cfg.optimizer)
            self.optimizer_G.load_state_dict(state_dict['Gen'])
            self.optimizer_D.load_state_dict(state_dict['Dis'])
        else:
            self.start_epoch = -1

        self.schedulers = []
        self.optimizers = []

        self.optimizers.append(self.optimizer_G)
        self.optimizers.append(self.optimizer_D)

        for optimizer in self.optimizers:
            self.schedulers.append(get_scheduler(optimizer, self.cfg, self.start_epoch))

    def compute_GAN_losses(self, net_D_output, dis_update):
        r"""Compute GAN loss and feature matching loss.

        Args:
            net_D_output (dict): Output of the discriminator.
            dis_update (bool): Whether to update discriminator.
        """
        if net_D_output['pred_fake'] is None:
            return torch.FloatTensor(1).fill_(0).to(self.device) if dis_update else [
                torch.FloatTensor(1).fill_(0).to(self.device),
                torch.FloatTensor(1).fill_(0).to(self.device)]
        if dis_update:
            # Get the GAN loss for real/fake outputs.
            GAN_loss = \
                self.criterionGAN(net_D_output['pred_fake']['output'], False,
                                     dis_update=True) + \
                self.criterionGAN(net_D_output['pred_real']['output'], True,
                                     dis_update=True)
            return GAN_loss
        else:
            # Get the GAN loss and feature matching loss for fake output.
            GAN_loss = self.criterionGAN(
                net_D_output['pred_fake']['output'], True, dis_update=False)

            FM_loss = self.criterion_matching(
                net_D_output['pred_fake']['features'],
                net_D_output['pred_real']['features'])
            return GAN_loss, FM_loss


    def set_input(self, data):

        # B * L * 3 * H * W -> L * B * 3 * H * W
        self.img  = data['img'].to(self.device).transpose(1,0)
        self.pose = data['pose'].to(self.device).transpose(1,0)
        self.skel = data['skel'].to(self.device).transpose(1,0)
        self.back = data['back'].to(self.device).transpose(1,0)
        self.label = torch.cat([self.skel, self.pose ], dim=2)

        # B * L * H * W -> L * B * H * W
        self.mask = data['mask'].to(self.device).detach().transpose(1,0)

    def forward(self):

        seq_len = self.img.shape[0]
        self.output_list = []
        self.gen_mask_list = []
        self.gen_img_list = []
        img_prev = None


        for i in range(seq_len-2):

            label = self.label [i+1]
            label_prev = self.label [i]

            img_back = self.back[i+1]
            img_prev = self.img[0] if img_prev is None else self.output_list[-1]

            real_img = self.img[i+1]
            fg_mask = self.mask[i+1].unsqueeze(1).repeat(1,3,1,1)

            gen, mask = self.net_G(label, label_prev, img_back, img_prev)

            self.gen_mask_list.append(mask.clone().detach())
            self.gen_img_list.append(gen.clone().detach())

            mask = mask.repeat(1,3,1,1)
            not_mask = (1 - mask)
            fuse = gen * mask + img_back * not_mask

            self.output_list.append(fuse.clone().detach())

            self.net_D_output = self.net_D(label, real_img, fuse.clone().detach(), \
                                           label_prev, img_prev, gen.clone().detach(), fg_mask)
            self.backward_D()

            self.net_D_output = self.net_D(label, real_img, fuse, label_prev, img_prev, gen, fg_mask)
            self.backward_G(fuse, real_img, fuse, gen, mask, fg_mask)



    def backward_D(self):

        self.loss_D = dict()

        self.optimizer_D.zero_grad()

        self.loss_D['fuse'] = self.compute_GAN_losses(self.net_D_output['indv'], dis_update=True)

        if 'raw' in self.net_D_output:
            self.loss_D['raw'] = self.compute_GAN_losses(self.net_D_output['raw'], dis_update=True)
        else:
            self.loss_D['raw'] = torch.FloatTensor(1).fill_(0).to(self.device)

        # Additional GAN loss.
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                self.loss_D[str(name)] = self.compute_GAN_losses(
                                    self.net_D_output[str(name)], dis_update=True)

        total_loss = torch.FloatTensor(1).fill_(0).to(self.device)

        for key in self.loss_D:
            total_loss += self.loss_D[key] * self.cfg.gan[key]
        
        self.loss_D['total'] = total_loss

        total_loss.backward()

        self.optimizer_D.step()


    def backward_G(self, final, real_img, fuse, gen, mask, fg_mask):

        self.loss_GAN_G = dict()
        self.loss_FM = dict()

        self.optimizer_G.zero_grad()

        self.loss_GAN_G['fuse'], self.loss_FM['fuse'] = \
            self.compute_GAN_losses(self.net_D_output['indv'], dis_update=False)

        if 'raw' in self.net_D_output:
            self.loss_GAN_G['raw'], self.loss_FM['raw'] = \
                self.compute_GAN_losses(self.net_D_output['raw'], dis_update=False)
        else:
            self.loss_GAN_G['raw'] = torch.FloatTensor(1).fill_(0).to(self.device)
            self.loss_FM['raw'] = torch.FloatTensor(1).fill_(0).to(self.device)

        # Additional GAN loss.
        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                self.loss_GAN_G[str(name)], self.loss_FM[str(name)] = \
                    self.compute_GAN_losses(self.net_D_output[str(name)], dis_update=False)

        ####

        self.loss_feature = self.criterion_perceptual(fuse, real_img) + \
                            self.criterion_perceptual(gen*fg_mask, real_img*fg_mask)
        self.loss_feature *= self.cfg.perceptual.weight

        ####

        self.loss_L1 = self.criterion_L1(fuse, real_img) + \
                       self.criterion_L1_masked(gen, fg_mask, real_img)
        self.loss_L1 *= self.cfg.l1_w

        ####

        self.loss_mask = self.mask_regulation(mask, fg_mask) * self.cfg.mask_w

        ####

        self.loss_GAN_G['total'] = torch.FloatTensor(1).fill_(0).to(self.device)
        for key in self.loss_GAN_G:
            if key != 'total':
               self.loss_GAN_G['total'] += self.loss_GAN_G[key] * self.cfg.gan[key]

        ####

        self.loss_FM['total'] = torch.FloatTensor(1).fill_(0).to(self.device)
        for key in self.loss_FM:
            if key != 'total':
                self.loss_FM['total'] += self.loss_FM[key] * self.cfg.fm_w

        self.loss_G = self.loss_GAN_G['total'] + \
                      self.loss_FM['total'] + \
                      self.loss_L1 + \
                      self.loss_feature + \
                      self.loss_mask

        self.loss_G.backward()
        self.optimizer_G.step()


    def optimize_parameters(self):
        self.forward()

    def update_learning_rate(self):
        for scheduler in self.schedulers:
           scheduler.step()

    def get_current_losses(self):
        losses = {}

        if self.add_dis_cfg is not None:
            for name in self.add_dis_cfg:
                losses['Dis/'+str(name) ] = self.loss_D[str(name)].data

        losses['Dis/fuse'] = self.loss_D['fuse'].data
        losses['Dis/raw'] = self.loss_D['raw'].data
        losses['Dis/total'] = self.loss_D['total'].data

        losses['Gen/loss_GAN_G'] = self.loss_GAN_G['total'].data
        losses['Gen/loss_FM'] = self.loss_FM['total'].data
        losses['Gen/loss_L1'] = self.loss_L1.data
        losses['Gen/loss_feature'] = self.loss_feature.data
        losses['Gen/mask'] = self.loss_mask.data
        losses['Gen/total'] = self.loss_G.data

        return losses

    def get_current_visuals(self, num_images=1):
        vis = {}

        vis['image/src'] = tensor2images (self.img[0].data, num_images)
        vis['image/gt'] = tensor2images (self.img[-2].data, num_images)
        vis['image/back'] = tensor2images (self.back[-2].data, num_images)
        vis['image/gen'] = tensor2images (self.gen_img_list[-1].data, num_images)
        vis['image/gen_mask'] = tensor2images (self.gen_mask_list[-1].data, num_images)
        vis['image/fuse'] = tensor2images (self.output_list[-1].data, num_images)

        vis['pose/src'] = tensor2images (self.skel[0].data, num_images)
        vis['pose/tar'] = tensor2images (self.skel[-2].data, num_images)
        vis['pose/mask'] = tensor2images (self.mask[-2].data.unsqueeze(1), num_images)

        return vis

    def save_network(self, epoch):
        
        save_path = os.path.join(self.out_path, "netG_epoch{:03d}.pth".format(epoch))
        torch.save(self.net_G.cpu().state_dict(), save_path)

        save_path = os.path.join(self.out_path, "netD_epoch{:03d}.pth".format(epoch))
        torch.save(self.net_D.cpu().state_dict(), save_path)

        # Don't save optimizer to save space
        #save_path = os.path.join(self.out_path, "opt_epoch{:03d}.pth".format(epoch))
        #torch.save({'Gen': self.optimizer_G.state_dict(), 'Dis': self.optimizer_D.state_dict() }, save_path)

        self.net_D.to(self.device)
        self.net_G.to(self.device)

