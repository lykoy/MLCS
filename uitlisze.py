import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import time, math, glob
import warnings
warnings.filterwarnings("ignore")
# import scipy.io as sio
#from skimage.measure import compare_ssim
import cv2
from skimage.metrics import structural_similarity as compare_ssim
# import imutils
# import os
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr


def testing(model,model_dir,recon_dir,test_dir):

    model.eval()
    model.load_state_dict(torch.load(model_dir))
    image_list = glob.glob(test_dir)
    save_dir = recon_dir
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    avg_elapsed_time = 0.0
    avg_loss = 0.0
    idx = 0
    with torch.no_grad():
        for image_name in image_list:
            #print("Processing ", image_name)
            image_test = Image.open(image_name)
            im_gray = image_test.convert('L')
            im_gt_y = np.array(im_gray).astype(float)

            im_input = im_gt_y/255.
            im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
            im_input = im_input.cuda()
            model = model.cuda()

            start_time = time.time()
            res = model(im_input)
            elapsed_time = time.time() - start_time
            avg_elapsed_time += elapsed_time
            res = res.cpu()

            im_res_y = res.data[0].numpy().astype(np.float32)

            im_res_y = im_res_y*255.
            im_res_y[im_res_y<0] = 0
            im_res_y[im_res_y>255.] = 255.
            im_res_y = im_res_y[0,:,:]

            im_recon_tensor = torch.from_numpy(im_res_y)
            im_orig_tensor = torch.from_numpy(im_gt_y)
            im_res_y = im_res_y.astype(np.uint8)
            im_gt_y = im_gt_y.astype(np.uint8)

            ssim_predicted = compare_ssim(im_gt_y, im_res_y, full=True) #ndarry int
            psnr_predicted = compare_psnr(im_gt_y, im_res_y) # int  #ndarry int
            loss = nn.SmoothL1Loss()
            l = loss(im_recon_tensor, im_orig_tensor) #tensor float

            avg_psnr_predicted += psnr_predicted
            avg_ssim_predicted += ssim_predicted[0]
            avg_loss += l

            idx += 1
            cv2.imwrite(save_dir + str(idx) + '.png', im_res_y)

    print("avg_PSNR=", avg_psnr_predicted/len(image_list))
    print("avg_SSIM=", avg_ssim_predicted/len(image_list))
    print("avg_loss=", avg_loss/len(image_list))
    print("It takes average {}s for processing".format(avg_elapsed_time/len(image_list)))
    return avg_psnr_predicted/len(image_list),avg_ssim_predicted/len(image_list),avg_loss/len(image_list)