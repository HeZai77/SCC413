#-*- coding: utf-8 -*-
import os
from argparse import ArgumentParser
import model as md
from utils import create_link
import test as tst


def get_args():
    parser = ArgumentParser(description='cycleGAN PyTorch')
    parser.add_argument('--epochs', type=int, default=100)                                              ### 训练 epoch 数量
    parser.add_argument('--decay_epoch', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--load_height', type=int, default=256)
    parser.add_argument('--load_width', type=int, default=200)
    parser.add_argument('--gpu_ids', type=str, default='-1', help='set -1 is cpu; set 0 is gpu')
    parser.add_argument('--crop_height', type=int, default=256)
    parser.add_argument('--crop_width', type=int, default=200)
    parser.add_argument('--lamda', type=int, default=1)
    parser.add_argument('--idt_coef', type=float, default=0.5)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--testing', type=bool, default=True)
    parser.add_argument('--results_dir', type=str, default='./results_gray')
    parser.add_argument('--dataset_dir', type=str, default='./datasets')
    parser.add_argument('--checkpoint_dir', type=str, default='./ckpt/weights/weights_resnet_9blocks_pair')
    parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
    parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--gen_net', type=str, default='resnet_9blocks')
    parser.add_argument('--dis_net', type=str, default='n_layers')
    parser.add_argument('--curr_epoch', type=int, default=140, help='load pretrain model id')
    args = parser.parse_args()
    return args


def main():
  args = get_args()

  str_ids = args.gpu_ids.split(',')
  args.gpu_ids = []

  for str_id in str_ids:
      id = int(str_id)
      if id >= 0:
          args.gpu_ids.append(id)

  if args.training:
      print("Training")
      model = md.cycleGAN(args)
      model.train(args)

  if args.testing:
      print("Testing")
      tst.test(args)


if __name__ == '__main__':
    main()