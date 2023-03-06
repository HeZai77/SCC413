# -*- encoding: utf-8 -*-
# -*- encoding: utf-8 -*-
from tqdm import tqdm
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import cv2
import lpips
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import argparse


def parse_opt():
    """ 参数设置 """
    parser = argparse.ArgumentParser(description='PSNR SSIM LPIPS', add_help=False)
    parser.add_argument('--gt_dir', default='./results_gray/gt')   ## 真实数据
    parser.add_argument('--input_dir', default='./results_gray/gen')   ## 生成数据
    parser.add_argument('-v', '--version', type=str, default='0.1')
    args = parser.parse_args()
    return args


def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg", ".png", ".jpeg"])


def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img / 255.
    return img


class DataLoaderVal(Dataset):
    def __init__(self, args, target_transform=None):
        super(DataLoaderVal, self).__init__()
        self.target_transform = target_transform
        gt_dir = args.gt_dir
        input_dir = args.input_dir
        clean_files = sorted(os.listdir(os.path.join(gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(input_dir)))
        self.clean_filenames = [os.path.join(gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(input_dir, x) for x in noisy_files if is_png_file(x)]
        self.tar_size = len(self.clean_filenames)

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]
        clean = clean.permute(2, 0, 1)
        noisy = noisy.permute(2, 0, 1)
        return clean, noisy, clean_filename, noisy_filename


if __name__ == '__main__':

    ## 参数解析
    args = parse_opt()

    ## 读取数据集
    test_dataset = DataLoaderVal(args=args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # ---------------------- PSNR + SSIM ----------------------
    psnr_val_rgb = []
    ssim_val_rgb = []
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        rgb_groundtruth = data_test[0].numpy().squeeze().transpose((1, 2, 0))
        rgb_restored = data_test[1]
        rgb_restored = torch.clamp(rgb_restored, 0, 1).cpu().numpy().squeeze().transpose((1, 2, 0))
        psnr_val_rgb.append(psnr(rgb_restored, rgb_groundtruth))
        ssim_val_rgb.append(ssim(rgb_restored, rgb_groundtruth, multichannel=True))

    psnr_val_rgb = sum(psnr_val_rgb) / len(test_dataset)
    ssim_val_rgb = sum(ssim_val_rgb) / len(test_dataset)

    # ---------------------- LPIPS ----------------------
    ## 初始化模型
    lpips_fun = lpips.LPIPS(net='alex', version=args.version)
    files = os.listdir(args.gt_dir)
    i = 0
    total_lpips_distance = 0
    average_lpips_distance = 0
    for file in files:
        try:
            ## 加载图像数据
            img0 = lpips.im2tensor(lpips.load_image(os.path.join(args.gt_dir, file)))
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(args.input_dir, file)))

            ## 计算指标
            current_lpips_distance = lpips_fun.forward(img0, img1)
            total_lpips_distance = total_lpips_distance + current_lpips_distance
            i = i + 1
        except Exception as e:
            print(e)

    average_lpips_distance = float(total_lpips_distance) / i

    print("The processed iamges is ", i)
    print("PSNR: %f, SSIM: %f, LPIPS: %f " % (psnr_val_rgb, ssim_val_rgb, average_lpips_distance))