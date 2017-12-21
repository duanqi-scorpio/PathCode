import numpy as np
import os
import sys
from PIL import Image
import scipy.ndimage as ndimage
from skimage.morphology import dilation
from skimage.filters import threshold_otsu
from skimage.morphology import star
from itertools import product
import random


import openslide

import app_logger
import file_api

# define the pixel meanings in mask image
BACKGROUND = 0
SELECTED = 1
NORMAL = 2
TUMOR = 3
global logger

class WSIProcessor(object):
    """The input data."""
    
    def __init__(self, setting):
        self._split_file = setting.split_file
        
        self._img_vis = setting.img_vis
        self._img_vis_patch = setting.img_vis_patch
        self._img_vis_patch_pos = os.path.join(self._img_vis_patch, 'pos')
        self._img_vis_patch_neg = os.path.join(self._img_vis_patch, 'neg')
        self._img_vis_patch_prob = setting.img_vis_patch_prob
        self._img_ext = setting.img_ext
        self._downsample_rate = setting.downsample_rate
        self._sample_upper_bound = setting.sample_upper_bound
        self._save_img = setting.save_img
        self._patch_size = setting.patch_size
        self._patch_factor = setting.patch_factor
        
        self._ov_size = int(np.ceil(self._patch_size / float(self._patch_factor)))

        if not os.path.exists(self._img_vis_patch_pos):
            os.makedirs(self._img_vis_patch_pos)
        if not os.path.exists(self._img_vis_patch_neg):
            os.makedirs(self._img_vis_patch_neg)

            
    def _threshold_downsample_lvl(self, lvl_img):
        """Generates thresholded overview image.
              
        Args: 
            wsi: An openslide image instance.

        Returns:
            A 2D numpy array of binary image
        """

        # calculate the overview level size and retrieve the image
        downsample_img = lvl_img.convert('HSV')
        downsample_img = np.array(downsample_img)

        # dilate image and then threshold the image 
        schannel = downsample_img[:, :, 1]
        schannel = dilation(schannel,star(3))
        schannel = ndimage.gaussian_filter(schannel, sigma=(5, 5), order=0)
        threshold_global = threshold_otsu(schannel)

        schannel[schannel > threshold_global] = 255
        schannel[schannel <= threshold_global] = 0

        #import scipy.misc   # check the result
        #scipy.misc.imsave('outfile.jpg', schannel) 

        return schannel 
        
    def _generate_mask(self, wsi, mask_files, svs_file, lvl_read):
        """Generates thresholded overview image.
              
        Args: 
            wsi: An openslide image instance.

        Returns:
            A 2D numpy array of binary image
        """
        logger = app_logger.get_logger('preprocess')
        self._wsi = wsi
        self._filename = svs_file
        # read the lowest WSI resolution
        lvl_dim = wsi.level_dimensions[lvl_read]
        #print svs_file, wsi.level_dimensions
        lvl_img = wsi.read_region((0,0), lvl_read, lvl_dim)
        lvl_threshold = self._threshold_downsample_lvl(lvl_img)
        
        # Initialize all pixels to background
        lvl_mask = np.zeros((lvl_dim[1], lvl_dim[0]), np.uint8)
        selected_mask = np.zeros((lvl_dim[1], lvl_dim[0]), np.uint8)
        # normal_mask = np.zeros((lvl_dim[1], lvl_dim[0]), np.uint8)
        tumor_mask = np.zeros((lvl_dim[1], lvl_dim[0]), np.uint8)
        assert lvl_mask.shape == (lvl_img.size[1], lvl_img.size[0])
        
        # fill in the corresponding region of the mask file
        if mask_files is not None:
            for mask_file in mask_files:
                # obtain the annotated region from annotation file name
                # anno = os.path.basename(mask_file)[: -5]. split('_')
                anno = file_api.get_mask_info(os.path.basename(mask_file.split('.')[0]))
                level = int(anno[1])
                origin = (int(anno[2]), int(anno[3]))
                size = (int(anno[4]), int(anno[5]))
                
                # read annotation file
                with open(mask_file, 'rb') as f:
                    mask_data = f.read()
                    mask_data = np.frombuffer(mask_data, np.uint8)
                    mask_data = mask_data.reshape([size[1], size[0]])
                
                new_origin = origin
                new_size = size
                factor = 1
                new_mask = Image.fromarray(mask_data)
                
                anno_lvl_size = wsi.level_dimensions[level]
                if (anno_lvl_size[0] == lvl_dim[0] and 
                        anno_lvl_size[1] == lvl_dim[1]):
                    pass
                elif (anno_lvl_size[0] != lvl_dim[0] and 
                        anno_lvl_size[1] != lvl_dim[1]):
                    factor = lvl_dim[0] / float(anno_lvl_size[0])
                    assert factor < 1
                    new_size = [int(np.ceil(size[0] * factor)),
                                int(np.ceil(size[1] * factor))]
                    new_origin = [int(np.ceil(origin[0] * factor)),
                                int(np.ceil(origin[1] * factor))]
                    new_mask = new_mask.resize(new_size)
                else:
                    logger.error('Error in WSI: {}'.format(svs_file))
                    
                # annotated region
                selected_mask[new_origin[1]: new_size[1] + new_origin[1],
                            new_origin[0]: new_size[0] + new_origin[0]] = 255
                new_mask = np.asarray(new_mask)
                #print mask_file, factor, wsi.level_dimensions, [new_size[1] + new_origin[1], new_size[0] + new_origin[0]]
                
                # if 'Normal' in svs_file:
                #     normal_mask[new_origin[1]: new_size[1] + new_origin[1],
                #                 new_origin[0]: new_size[0] + new_origin[0]] = new_mask
                # else:
                #     tumor_mask[new_origin[1]: new_size[1] + new_origin[1],
                #             new_origin[0]: new_size[0] + new_origin[0]] = new_mask
                tumor_mask[new_origin[1]: new_size[1] + new_origin[1],
                            new_origin[0]: new_size[0] + new_origin[0]] = new_mask
            lvl_mask[selected_mask != 0] = SELECTED
        
            normal_tissue_and = np.logical_and(lvl_threshold, selected_mask)
            #print (len(normal_tissue_and != 0))
            #print len(tumor_mask != 0)
            lvl_mask[normal_tissue_and != 0] = NORMAL
            lvl_mask[tumor_mask != 0] = TUMOR
            
        else:
            lvl_mask[lvl_threshold != 0] = NORMAL
        
        # lvl_mask[selected_mask != 0] = SELECTED
        
        # if 'Normal' in svs_file:
        #     lvl_mask[normal_mask != 0] = NORMAL
        # else:
        #     normal_tissue_and = np.logical_and(lvl_threshold, selected_mask)
        #     #print (len(normal_tissue_and != 0))
        #     #print len(tumor_mask != 0)
        #     lvl_mask[normal_tissue_and != 0] = NORMAL
        #     lvl_mask[tumor_mask != 0] = TUMOR
        
        wsi_dim = wsi.level_dimensions[0]
        new_size = (wsi_dim[0] / self._patch_factor, wsi_dim[1] / self._patch_factor)
        if wsi_dim[0] / lvl_dim[0] < self._patch_factor:
            lvl_mask = Image.fromarray(lvl_mask)
            lvl_mask = lvl_mask.resize(new_size)
            lvl_mask = np.asarray(lvl_mask)
            #print svs_file
        elif wsi_dim[0] / lvl_dim[0] > self._patch_factor:
            logger.error('Need a larger patch factor')
            
        if self._save_img:
            # save the overview level to check the processing is correct
            save_size = (int(np.ceil(self._downsample_rate * lvl_dim[0])),
                        int(np.ceil(self._downsample_rate * lvl_dim[1])))
            save_img = lvl_img.resize(save_size)
            save_img.save(os.path.join(self._img_vis, 
                          os.path.basename(svs_file)[:-4] + self._img_ext))
                          
            #lvl_threshold = Image.fromarray(lvl_threshold)
            #lvl_threshold.save(os.path.join(self._img_vis, 
            #                   os.path.basename(svs_file)[:-4] + '_mask1.png'))
            
            lvl_mask_save = np.zeros((new_size[1], new_size[0], 3), np.uint8)
            lvl_mask_save[lvl_mask == TUMOR] = [255, 0, 0]
            lvl_mask_save[lvl_mask == NORMAL] = [0, 255, 0]
            lvl_mask_save[lvl_mask == SELECTED] = [0, 0, 255]
            lvl_mask_save = Image.fromarray(lvl_mask_save)
            #save_img = lvl_mask_save.resize(save_size)
            lvl_mask_save.save(os.path.join(self._img_vis, 
                               os.path.basename(svs_file)[:-4] + '_mask' + self._img_ext))
                          
            save_img.close()
            lvl_mask_save.close()
            
        # close Image 
        lvl_img.close()
        return lvl_mask
    
    def _read_lowest_reso_lvls(self, mask_files):
        lvls = []
        for mask_file in mask_files:
            lvl = file_api.get_mask_info(os.path.basename(mask_file))[1]
            lvls.append(int(lvl))
        lvls = np.array(lvls)
        return lvls.max()

    def _save_random_patch(self, ov_patch, origin, f_type):

        if random.random()>self._img_vis_patch_prob:
            return

        img = self._wsi.read_region(origin, 0, (self._patch_size, self._patch_size))

        # resize mask
        mask = Image.fromarray(ov_patch)
        mask = mask.resize(img.size)
        mask = np.array(mask)

        img = np.array(img)

        alpha = 0.3
        if (mask == TUMOR).any():
            img[mask == TUMOR] = (1 - alpha) * img[mask == TUMOR] + alpha * np.array(
                [255, 0, 0])
        if (mask == NORMAL).any():
            img[mask == NORMAL] = (1 - alpha) * img[mask == NORMAL] + alpha * np.array(
                [0, 255, 0])
        if (mask == SELECTED).any():
            img[mask == SELECTED] = (1 - alpha) * img[mask == SELECTED] + alpha * np.array(
                [0, 0, 255])

        img = Image.fromarray(img)
        if f_type=='pos':
            img.save(os.path.join(self._img_vis_patch_pos,
                                        os.path.basename(self._filename).split('.')[0]
                                            + '_%d_%d'%( origin[0], origin[1])+ self._img_ext))
        else:
            img.save(os.path.join(self._img_vis_patch_neg,
                                    os.path.basename(self._filename).split('.')[0]
                                        + '_%d_%d'%( origin[0], origin[1])+ self._img_ext))
        img.close()
    #################################################################
    def _get_training_tiles(self, lvl_mask):
        num_row, num_col = lvl_mask.shape
        num_row = num_row - self._ov_size
        num_col = num_col - self._ov_size
        
        label_dict = {'pos': [], 'neg': []}
        threshold = self._ov_size * self._ov_size * 0.5

        row_col = list(product(range(num_row), range(num_col)))
        random.shuffle(row_col)
        cnt = 0
        # for row, col in product(range(num_row), range(num_col)):
        for row, col in row_col:

            if cnt>self._sample_upper_bound+1000:
                break

            ov_patch = lvl_mask[row: row + self._ov_size,
                                col: col + self._ov_size]
            # if np.count_nonzero(ov_patch == NORMAL) > threshold:
            #     origin = (col * self._patch_factor, row * self._patch_factor)
            #     label_dict['neg'].append(origin)
            # elif np.count_nonzero(ov_patch == TUMOR) > threshold:
            #     origin = (col * self._patch_factor, row * self._patch_factor)
            #     label_dict['pos'].append(origin)

            origin = (col * self._patch_factor, row * self._patch_factor)
            # img = self._wsi.read_region(origin, 0, (self._patch_size, self._patch_size))
            # # bad case is background    continue
            # if np.array(img)[:, :, 1].mean() > 200:
            #     continue
            # img.close()

            H, W = ov_patch.shape
            H_min = int(np.ceil(H/4))
            H_max = int(np.ceil(H/4*3))
            W_min = int(np.ceil(W/4))
            W_max = int(np.ceil(W/4*3))
            if np.count_nonzero(ov_patch[H_min:H_max, W_min:W_max] == TUMOR)>0:
                if self._type == 'pos':
                    # origin = (col * self._patch_factor, row * self._patch_factor)
                    img = self._wsi.read_region(origin, 0, (self._patch_size, self._patch_size))
                    # bad case is background    continue
                    if np.array(img)[:, :, 1].mean() > 200:
                        img.close()
                        continue
                    img.close()

                    label_dict['pos'].append(origin)
                    self._save_random_patch(ov_patch, origin, 'pos')
                    cnt+=1

            else:
                if self._type == 'neg':
                    # origin = (col * self._patch_factor, row * self._patch_factor)
                    img = self._wsi.read_region(origin, 0, (self._patch_size, self._patch_size))
                    # bad case is background    continue
                    if np.array(img)[:, :, 1].mean() > 200:
                        img.close()
                        continue
                    img.close()

                    label_dict['neg'].append(origin)
                    self._save_random_patch(ov_patch, origin, 'neg')
                    cnt+=1

        return label_dict
    
    def _adjust_level(self, wsi, level_idx):
        while wsi.level_dimensions[level_idx][0]<2048 and\
                        wsi.level_dimensions[level_idx][1]<2048:
            level_idx-=1
        return level_idx

    def process(self, item, type):
        logger = app_logger.get_logger('preprocess')
        self._type = type
        raw_file = item['WSI'][0]
        wsi = file_api.AllSlide(raw_file)
        level_idx = wsi.level_count - 1
        
        mask_files = item['WSI'][1]

        if mask_files is None:
            lowest_mask_lvl = level_idx
        else:
            lowest_mask_lvl = self._read_lowest_reso_lvls(mask_files)
        if level_idx < lowest_mask_lvl:
            logger.error('Annotation for {} is Wrong'.format(raw_file))

        level_idx = self._adjust_level(wsi, level_idx)

        lvl_mask = self._generate_mask(wsi, mask_files, raw_file, level_idx)
        
        #factor = wsi.level_dimensions[0][0] / lvl_mask.shape[1]
        #print factor
        return self._get_training_tiles(lvl_mask)
        
        
