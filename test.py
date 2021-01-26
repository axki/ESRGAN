import os.path as osp
import glob
import cv2
import numpy as np
import torch
import RRDBNet_arch as arch
import metrics as es
import sys
from tqdm import tqdm

device = torch.device('cuda')

print(sys.argv)
scale = 4
test_dir = sys.argv[1]
model_path = sys.argv[2]
test_hr_dir = osp.join(test_dir, 'HR/')
test_lr_dir = osp.join(test_dir, 'LR/', f'X{scale}')
hr_dirlist = sorted(glob.glob(test_hr_dir+'/*'))
lr_dirlist = sorted(glob.glob(test_lr_dir+'/*'))

model = arch.RRDBNet(3, 3, 64, 23, gc=32)
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model = model.to(device)

print('Model path {:s}. \nTesting...'.format(model_path))

mlog = {'psnr':[], 'ssim':[], 'mse':[]}  # (self)
idx = 0
for hr_path, lr_path in tqdm(zip(hr_dirlist, lr_dirlist), total=len(lr_dirlist)):
    idx += 1
    img = cv2.imread(lr_path, cv2.IMREAD_COLOR)
    img = img * 1.0 / 255
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img_LR = img.unsqueeze(0)
    img_LR = img_LR.to(device)

    with torch.no_grad():
        output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
    output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round()
    
    # (self): perform metrics calculation
    hr = cv2.imread(hr_path, cv2.IMREAD_COLOR)
    hr = hr.reshape((hr.shape[2], hr.shape[0], hr.shape[1]))
    sr = output.reshape((output.shape[2], output.shape[0], output.shape[1]))
    assert hr.shape == sr.shape, f"shape not the same, hr: {hr.shape}, sr: {sr.shape}"
    mlog['psnr'].append(es.get_metric(hr, sr, 'psnr'))
    mlog['ssim'].append(es.get_metric(hr, sr, 'ssim'))
    mlog['mse'].append(es.get_metric(hr, sr, 'ssim'))
    
    cv2.imwrite('output/{:d}_rlt.png'.format(idx-1), output)
print(f"max psnr: {np.max(mlog['psnr'])}, mean psnr: {np.mean(mlog['psnr'])}")
print(f"max ssim: {np.max(mlog['ssim'])}, mean ssim: {np.mean(mlog['ssim'])}")
print(f"max mse: {np.max(mlog['mse'])}, mean mse: {np.mean(mlog['mse'])}")
