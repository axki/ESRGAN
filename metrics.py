import cv2
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
import numpy as np

def get_y_component(rgb):
    # cv2 assumes color channels is last
    rgb  = rgb.reshape(rgb.shape[1], rgb.shape[2], rgb.shape[0])
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    return ycrcb[:, :, 0]

def get_metric(hr, sr, metric='psnr'):
    hr_y = get_y_component(hr)
    sr_y = get_y_component(sr)
    if metric == 'psnr':
        res = psnr(hr_y, sr_y)
    elif metric == 'ssim':
        res = ssim(hr_y, sr_y)
    elif metric == 'mse':
        res = mse(hr_y, sr_y)
    elif metric == 'all':
        res = (psnr(hr_y, sr_y), ssim(hr_y, sr_y), mse(hr_y, sr_y))
    else:
        raise NotImplementedError()
    return res

def psnr(hr, sr):
    ''' assumes 1ch Y-component input (from YCrCb)'''
    return peak_signal_noise_ratio(hr, sr, data_range=sr.max()-sr.min())
    
def mse(hr, sr):
    ''' assumes 1ch Y-component input (from YCrCb)'''
    return mean_squared_error(hr, sr)

def ssim(hr, sr):
    ''' assumes 1ch Y-component input (from YCrCb)'''
    return structural_similarity(hr, sr, data_range=sr.max()-sr.min(), gaussian_weights=True)







