import numpy as np
import os
from PIL import Image
import scipy.ndimage as ndimage
from skimage.morphology import dilation
from skimage.filters import threshold_otsu
from skimage.morphology import star
from itertools import product
import glob
import random
import scipy.misc

import file_api

import app_logger

def _adjust_level(wsi, level_idx):
        while wsi.level_dimensions[level_idx][0]<2048 and\
                        wsi.level_dimensions[level_idx][1]<2048:
            level_idx-=1
        return level_idx

Normal_WSI_dir = '/home/users/duanqi01/PathData/RJPathData/Normal/'
output_dir = '/home/users/duanqi01/PathData/RJPathData/Normal_anno/'
img_type = ['*.svs', '*.kfb']

wsi_filelist = []
for t in img_type:
    wsi_filelist.extend(glob.glob(os.path.join(Normal_WSI_dir, t)))

print len(wsi_filelist)

target_level = 0

for file in wsi_filelist:
    wsi_filename = os.path.basename(file)
    print wsi_filename
    wsi= file_api.AllSlide(file)
    level_idx = wsi.level_count - 1
    level_idx = _adjust_level(wsi, level_idx)

    lvl_dim = wsi.level_dimensions[level_idx]
    lvl_img = wsi.read_region((0,0), level_idx, lvl_dim)
    lvl_img.save(output_dir+wsi_filename.split('.')[0]+'_thumb.jpeg')

    t_img = np.array(lvl_img.convert('HSV'))
    schannel = t_img[:,:,1]
    schannel = dilation(schannel,star(3))
    schannel = ndimage.gaussian_filter(schannel, sigma=(5, 5), order=0)
    threshold_global = threshold_otsu(schannel)
    schannel[schannel > threshold_global] = 255
    schannel[schannel <= threshold_global] = 0
    scipy.misc.imsave(output_dir+wsi_filename.split('.')[0]+'_all.jpeg', schannel)

    # f = open(output_dir+wsi_filename.split('.')[0]+'.mask','wb')
    # f.write(schannel)
    # f.close()
    #
    # with open(output_dir+wsi_filename.split('.')[0]+'.mask', 'rb') as f:
    #     mask_data = f.read()
    #     mask_data = np.frombuffer(mask_data, np.uint8)
    #     mask_data = mask_data.reshape([lvl_dim[1], lvl_dim[0]])
    #     scipy.misc.imsave(output_dir+wsi_filename.split('.')[0]+'_afterregion.jpeg', mask_data)

    tar_dim = wsi.level_dimensions[target_level]
    tmp_channel = scipy.misc.imresize(schannel, (tar_dim[1],tar_dim[0]))
    src_x = int(tar_dim[0]/4)
    src_y = int(tar_dim[1]/4)
    tar_width = int(tar_dim[0]/2)
    tar_height = int(tar_dim[1]/2)
    tar_channel = tmp_channel[src_y:src_y+tar_height, src_x:src_x+tar_width].copy()
    #scipy.misc.imsave(output_dir + wsi_filename.split('.')[0] + '_&_'+str(target_level)+'_&_'+str(src_x)+'_&_'+str(src_y)+'_&_'+str(tar_width)+'_&_'+str(tar_height)+'_center.jpeg', tar_channel)
    f = open(output_dir + wsi_filename.split('.')[0] + '_&_'+str(target_level)+'_&_'+str(src_x)+'_&_'+str(src_y)+'_&_'+str(tar_width)+'_&_'+str(tar_height)+'.mask', 'wb')
    f.write(tar_channel)
    f.close()
    print output_dir + wsi_filename.split('.')[0] + '_&_'+str(target_level)+'_&_'+str(src_x)+'_&_'+str(src_y)+'_&_'+str(tar_width)+'_&_'+str(tar_height)+'.mask'
    # with open(output_dir + wsi_filename.split('.')[0] + '_&_'+str(target_level)+'_&_'+str(src_x)+'_&_'+str(src_y)+'_&_'+str(tar_width)+'_&_'+str(tar_height)+'.mask', 'rb') as f:
    #     mask_data = f.read()
    #     mask_data = np.frombuffer(mask_data, np.uint8)
    #     mask_data = mask_data.reshape([tar_height, tar_width])
    #     scipy.misc.imsave(output_dir + wsi_filename.split('.')[0] + '_&_'+str(target_level)+'_&_'+str(src_x)+'_&_'+str(src_y)+'_&_'+str(tar_width)+'_&_'+str(tar_height)+'_afteranno.jpeg', mask_data)









