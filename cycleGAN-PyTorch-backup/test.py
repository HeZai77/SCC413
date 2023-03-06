#-*- coding:utf-8 -*-
import os, sys
import torch
from torch import nn
from torch.autograd import Variable
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import utils
from arch import define_Gen
from datasets import SubImageDataset, ImageDataset
from collections import OrderedDict
from skimage.metrics import structural_similarity as ssim
import numpy as np
import cv2


def test(args):
    ## 读入图像数据进行预处理
    transform = transforms.Compose(
        [transforms.Resize((args.crop_height, args.crop_width)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5,0.5,0.5,], std=[0.5,0.5,0.5,])])

    ## 获取测试数据的文件路径
    dataset_dirs = utils.get_testdata_link(args.dataset_dir)

    ## 以dataloader方式读取测试数据, batch_size=1
    a_dataset = SubImageDataset(root=dataset_dirs['testA'], transforms_=transform, dataname="A")
    a_loader = torch.utils.data.DataLoader(a_dataset, batch_size=1, shuffle=False, num_workers=0)

    b_dataset = SubImageDataset(root=dataset_dirs['testB'], transforms_=transform, dataname="B")
    b_loader = torch.utils.data.DataLoader(b_dataset, batch_size=1, shuffle=False, num_workers=0)

    ## 读取成对数据
    dataset = ImageDataset(root=args.dataset_dir, transforms_=transform, phase="test")
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    ## 初始化生成网络，分别为b->a, a->b
    Gab = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)
    Gba = define_Gen(input_nc=3, output_nc=3, ngf=args.ngf, netG=args.gen_net, norm=args.norm,
                                                    use_dropout= not args.no_dropout, gpu_ids=args.gpu_ids)

    ## 打印输出网络参数量
    utils.print_networks([Gab, Gba], ['Gab','Gba'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ## 加载网络权重模型 (训练模型网络为torch.nn.DataParallel, 测试在cpu上，所以需要去除权重文件key字段中的 module)
    try:
        ckpt = utils.load_checkpoint('%s/latest_%d.ckpt' % (args.checkpoint_dir, args.curr_epoch), map_location=device)
        print(ckpt.keys())
        Gab_ckpt = ckpt['Gab']
        Gab_state_dict = OrderedDict()
        for k, v in Gab_ckpt.items():
            name = k.replace('module.', '')
            Gab_state_dict[name] = v

        Gba_ckpt = ckpt['Gba']
        Gba_state_dict = OrderedDict()
        for k, v in Gba_ckpt.items():
            name = k.replace('module.', '')
            Gba_state_dict[name] = v

        Gab.load_state_dict(Gab_state_dict)
        Gba.load_state_dict(Gba_state_dict)

    except Exception as e:
        print(e)
        print(' [*] No checkpoint!')
        sys.exit(0)

    Gab.eval()
    Gba.eval()

    ## 保存生成图片的位置
    if not os.path.isdir(args.results_dir):
        os.makedirs(args.results_dir)

    ## 保存真实数据的文件夹
    save_gt_dir = os.path.join(args.results_dir, "gt")
    os.makedirs(save_gt_dir, exist_ok=True)
    ## 保存生成数据的文件夹
    save_gen_dir = os.path.join(args.results_dir, "gen")
    os.makedirs(save_gen_dir, exist_ok=True)

    idx = 0
    avg_ssim = 0
    # for a_real_test, b_real_test in zip(a_loader, b_loader):
    for i, (a_real_test, b_real_test) in enumerate(loader):
        idx += 1
        ## 读取待输入网络数据
        # a_real_test = Variable(iter(a_loader).next(), requires_grad=False)
        # b_real_test = Variable(iter(b_loader).next(), requires_grad=False)

        a_real_test, b_real_test = utils.cuda([a_real_test, b_real_test])

        ## 网络forward, 生成域转换后的新数据
        with torch.no_grad():
            a_fake_test = Gab(b_real_test)
            b_fake_test = Gba(a_real_test)
            a_recon_test = Gab(b_fake_test)
            b_recon_test = Gba(a_fake_test)

        pic = (torch.cat([a_real_test, b_fake_test, b_real_test, a_fake_test], dim=0).data + 1) / 2.0

        ## 保存图片数据 (保存的数据有，A真实->A生成->A重建；B真实->B生成->B重建)
        torchvision.utils.save_image(pic, args.results_dir+'/sample_%d_epoch%d.jpg'%(idx, args.curr_epoch), nrow=2)

        ## 只保存A生成
        # print("a_fake_test", torch.max(a_fake_test.data), torch.min(a_fake_test.data))
        pic_A_fake = (a_fake_test.data + 1) / 2.0
        torchvision.utils.save_image(pic_A_fake, "a_fake.jpg")

        ## 只保存B生成
        pic_B_fake = (b_fake_test.data + 1) / 2.0
        torchvision.utils.save_image(pic_B_fake, "b_fake.jpg")

        ## 保存真实结果和生成结果
        pic_B = (torch.cat([b_real_test, b_fake_test], dim=0).data + 1) / 2.0
        torchvision.utils.save_image(pic_B, args.results_dir + '/sample_%d_test.jpg' % idx, nrow=2)

        ## 计算结构相似性
        pic_B_real = ((b_real_test.data + 1) / 2.0) * 255.0
        pic_B_fake = ((b_fake_test.data + 1) / 2.0) * 255.0
        # print(pic_B_fake.shape, pic_B_real.shape)
        pic_B_real = pic_B_real.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
        pic_B_fake = pic_B_fake.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8)
        # cv2.imshow("show_real", pic_B_real)
        # cv2.imshow("show_fake", pic_B_fake)

        ## 保存真实图和生成图，名字完全一致
        cv2.imwrite("{}/sample_{}.jpg".format(save_gt_dir, str(idx)), pic_B_real)
        cv2.imwrite("{}/sample_{}.jpg".format(save_gen_dir, str(idx)), pic_B_fake)

        # # pic_B_real = cv2.cvtColor(pic_B_real, cv2.COLOR_RGB2GRAY)
        # # pic_B_fake = cv2.cvtColor(pic_B_fake, cv2.COLOR_RGB2GRAY)
        # score_ssim = ssim(pic_B_real, pic_B_fake, multichannel=True)
        # print("score_ssim: ", score_ssim)
        # avg_ssim += score_ssim

    # print("avg_ssim: ", avg_ssim / idx)
    # return avg_ssim / idx

