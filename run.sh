# only x4 scale pretrained models available

# Eval RRDB ESRGAN x4
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/PD_2D_COR models/RRDB_ESRGAN_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/PD_2D_SAG models/RRDB_ESRGAN_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/PD_2D_TRA models/RRDB_ESRGAN_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/T1_2D_SAG models/RRDB_ESRGAN_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/T1_VIBE_2D_SAG models/RRDB_ESRGAN_x4.pth
echo '---'

# Eval RRGB ESRGAN x4 optimized for high PSNR
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/PD_2D_COR models/RRDB_PSNR_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/PD_2D_SAG models/RRDB_PSNR_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/PD_2D_TRA models/RRDB_PSNR_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/T1_2D_SAG models/RRDB_PSNR_x4.pth
CUDA_VISIBLE_DEVICES=1 python3 test.py /opt/temp/png_div2k/T1_VIBE_2D_SAG models/RRDB_PSNR_x4.pth
echo '---'
