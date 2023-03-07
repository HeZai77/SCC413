#-*- coding: utf-8 -*-
import itertools
import functools

import os
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import utils
from arch import define_Gen, define_Dis, set_grad
from torch.optim import lr_scheduler
from datasets import ImageDataset, SubImageDataset
from arch.loss import PerceptualLoss
import test as tst


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028
    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss
    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            alpha = alpha.to(device)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


class cycleGAN(object):
    def __init__(self, args):
        # Define the network 
        #####################################################
        self.Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm, use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
        self.Da = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)
        self.Db = define_Dis(input_nc=3, ndf=args.ndf, netD= args.dis_net, n_layers_D=3, norm=args.norm, gpu_ids=args.gpu_ids)

        utils.print_networks([self.Gab,self.Gba,self.Da,self.Db], ['Gab','Gba','Da','Db'])

        # Define Loss criterias
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.criterionPercep = PerceptualLoss()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Optimizers
        #####################################################
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gab.parameters(),self.Gba.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Da.parameters(),self.Db.parameters()), lr=args.lr, betas=(0.5, 0.999))

        ## new_lr = init_lr * lr_lambda
        self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)
        self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer, lr_lambda=utils.LambdaLR(args.epochs, 0, args.decay_epoch).step)

        # Try loading checkpoint
        #####################################################
        if not os.path.isdir(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

        try:
            ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
            self.start_epoch = ckpt['epoch']
            self.Da.load_state_dict(ckpt['Da'])
            self.Db.load_state_dict(ckpt['Db'])
            self.Gab.load_state_dict(ckpt['Gab'])
            self.Gba.load_state_dict(ckpt['Gba'])
            self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
            self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        except:
            print(' [*] No checkpoint!')
            self.start_epoch = 0


    def train(self,args):
        # For transforming the input image
        transformA = transforms.Compose([#transforms.RandomHorizontalFlip(),
                                         transforms.Resize((args.load_height, args.load_width)),
                                         transforms.RandomCrop((args.crop_height, args.crop_width)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        transformB = transforms.Compose([  # transforms.RandomHorizontalFlip(),
            transforms.Resize((args.load_height, args.load_width)),
            transforms.RandomCrop((args.crop_height, args.crop_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

        dataset_dirs = utils.get_traindata_link(args.dataset_dir)
        print("dataset_dirs: ", dataset_dirs)

        ## load unpair data
        a_dataset = SubImageDataset(root=dataset_dirs['trainA'], transforms_=transformA, dataname="A")
        a_loader = torch.utils.data.DataLoader(a_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        b_dataset = SubImageDataset(root=dataset_dirs['trainB'], transforms_=transformB, dataname="B")
        b_loader = torch.utils.data.DataLoader(b_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        ## load pair data
        dataset = ImageDataset(root=args.dataset_dir, transforms_=transformA)
        loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

        a_fake_sample = utils.Sample_from_Pool()
        b_fake_sample = utils.Sample_from_Pool()

        for epoch in range(self.start_epoch, args.epochs):

            lr = self.g_optimizer.param_groups[0]['lr']
            print('learning rate = %.7f' % lr)

            # for i, (a_real, b_real) in enumerate(zip(a_loader, b_loader)):
            for i, (a_real, b_real) in enumerate(loader):

                for j in range(1):
                    # Generator Computations
                    ##################################################
                    set_grad([self.Da, self.Db], False)
                    self.g_optimizer.zero_grad()

                    a_real = Variable(a_real)
                    b_real = Variable(b_real)
                    a_real, b_real = utils.cuda([a_real, b_real])

                    # Forward pass through generators
                    ##################################################
                    a_fake = self.Gab(b_real)
                    b_fake = self.Gba(a_real)

                    a_recon = self.Gab(b_fake)
                    b_recon = self.Gba(a_fake)

                    a_idt = self.Gab(a_real)
                    b_idt = self.Gba(b_real)

                    # Identity losses
                    ###################################################
                    a_idt_loss = self.L1(a_idt, a_real) * args.lamda * args.idt_coef
                    b_idt_loss = self.L1(b_idt, b_real) * args.lamda * args.idt_coef

                    # Adversarial losses
                    ###################################################
                    a_fake_dis = self.Da(a_fake)
                    b_fake_dis = self.Db(b_fake)

                    real_label = utils.cuda(Variable(torch.ones(a_fake_dis.size())))

                    a_gen_loss = self.MSE(a_fake_dis, real_label)
                    b_gen_loss = self.MSE(b_fake_dis, real_label)

                    # Cycle consistency losses
                    ###################################################
                    a_cycle_loss = self.L1(a_recon, a_real) * args.lamda
                    b_cycle_loss = self.L1(b_recon, b_real) * args.lamda
                    # g_percep_loss = self.criterionPercep((a_fake + 1.) / 2., (a_real + 1.) / 2.)

                    # Total generators losses
                    ###################################################
                    gen_loss = a_gen_loss + b_gen_loss + a_cycle_loss + b_cycle_loss + a_idt_loss + b_idt_loss

                    # Update generators
                    ###################################################
                    gen_loss.backward()
                    self.g_optimizer.step()

                a_real_vis = a_real[:2]
                b_real_vis = b_real[:2]
                a_fake_vis = a_fake[:2]
                b_fake_vis = b_fake[:2]
                a_recon_vis = a_recon[:2]
                b_recon_vis = b_recon[:2]

                # Discriminator Computations
                #################################################
                set_grad([self.Da, self.Db], True)
                self.d_optimizer.zero_grad()

                # Sample from history of generated images
                #################################################
                a_fake = Variable(torch.Tensor(a_fake_sample([a_fake.cpu().data.numpy()])[0]))
                b_fake = Variable(torch.Tensor(b_fake_sample([b_fake.cpu().data.numpy()])[0]))
                a_fake, b_fake = utils.cuda([a_fake, b_fake])

                # Forward pass through discriminators
                ################################################# 
                a_real_dis = self.Da(a_real)
                a_fake_dis = self.Da(a_fake)
                b_real_dis = self.Db(b_real)
                b_fake_dis = self.Db(b_fake)
                a_real_label = utils.cuda(Variable(torch.ones(a_real_dis.size())))
                a_fake_label = utils.cuda(Variable(torch.zeros(a_fake_dis.size())))
                b_real_label = utils.cuda(Variable(torch.ones(b_real_dis.size())))
                b_fake_label = utils.cuda(Variable(torch.zeros(b_fake_dis.size())))

                # Discriminator losses
                ##################################################
                a_dis_real_loss = self.MSE(a_real_dis, a_real_label)
                a_dis_fake_loss = self.MSE(a_fake_dis, a_fake_label)
                b_dis_real_loss = self.MSE(b_real_dis, b_real_label)
                b_dis_fake_loss = self.MSE(b_fake_dis, b_fake_label)

                # ## Discriminator wgan loss
                # cal_gradient_penalty(netD=self.Da, real_data=a_real, fake_data=a_fake, device=self.device)
                # cal_gradient_penalty(netD=self.Db, real_data=b_real, fake_data=b_fake, device=self.device)

                # Total discriminators losses
                a_dis_loss = (a_dis_real_loss + a_dis_fake_loss)*0.5
                b_dis_loss = (b_dis_real_loss + b_dis_fake_loss)*0.5

                # Update discriminators
                ##################################################
                a_dis_loss.backward()
                b_dis_loss.backward()
                self.d_optimizer.step()

                print("Epoch: (%3d) (%5d/%5d) | Gen Loss:%.2e | Dis Loss:%.2e" % 
                                            (epoch, i + 1, min(len(a_loader), len(b_loader)),
                                                            gen_loss, a_dis_loss+b_dis_loss))

            pic = (torch.cat([a_real_vis, b_fake_vis, a_recon_vis, b_real_vis, a_fake_vis, b_recon_vis], dim=0).data + 1) / 2.0
            if not os.path.isdir(args.checkpoint_dir):
                os.makedirs(args.checkpoint_dir, exist_ok=True)
            torchvision.utils.save_image(pic, args.checkpoint_dir + '/sample_epoch%d.jpg' % (args.curr_epoch), nrow=3)

            # Override the latest checkpoint
            #######################################################
            utils.save_checkpoint({'epoch': epoch + 1,
                                   'Da': self.Da.state_dict(),
                                   'Db': self.Db.state_dict(),
                                   'Gab': self.Gab.state_dict(),
                                   'Gba': self.Gba.state_dict(),
                                   'd_optimizer': self.d_optimizer.state_dict(),
                                   'g_optimizer': self.g_optimizer.state_dict()},
                                  '%s/latest_%d.ckpt' % (args.checkpoint_dir, epoch))

            # Update learning rates
            ########################
            self.g_lr_scheduler.step()
            self.d_lr_scheduler.step()
