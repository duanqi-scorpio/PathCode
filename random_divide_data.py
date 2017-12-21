import os
import glob
import sys
import ConfigParser
import numpy as np
from random import shuffle
import json

import openslide

import app_logger
import file_api
img_type = ['*.svs', '*.kfb']

class SJPathSetting(object):
    def __init__(self):
        pass
        
    def parse(self, config):
        # data folders
        self.normal_wsi_folder = config.get('DEFAULT', 'normal_wsi_folder')
        # self.normal_anno_folder = config.get('DEFAULT', 'normal_anno_folder')
        self.tumor_wsi_folder = config.get('DEFAULT', 'tumor_wsi_folder')
        self.tumor_anno_folder = config.get('DEFAULT', 'tumor_anno_folder')
        self.test_frac = config.getfloat('DEFAULT', 'test_frac')
        self.val_normal = config.getint('DEFAULT', 'val_normal')
        self.val_tumor = config.getint('DEFAULT', 'val_tumor')
        self.split_file = config.get('DEFAULT', 'split_file')
        self.split_folder = config.get('DEFAULT', 'split_folder')
        self.test_file = config.get('DEFAULT', 'test_file')
        
        if not os.path.exists(self.split_folder):
            os.makedirs(self.split_folder)

def _split_dataset(files, test_frac, val_num):
    num_imgs = len(files)
    num_selected = int(np.floor(test_frac * num_imgs))
    img_indices = range(num_imgs)
    shuffle(img_indices)
    
    train_files = []
    test_files = []
    val_files = []
    for idx in img_indices[num_selected: num_imgs - val_num]:
        train_files.append({'WSI': files[idx], 'split': None})
    
    for idx in img_indices[num_imgs - val_num:]:
        val_files.append({'WSI': files[idx], 'split': None})
    
    for idx in img_indices[: num_selected]:
        test_files.append({'WSI': files[idx], 'split': None})
    return train_files, test_files, val_files
    
def _remove_corrupted_files(files):
    correct_files = []
    for item in files:
        svs_file = item[0]
        try:
            wsi = file_api.AllSlide(svs_file)
            level = wsi.level_count - 1
            level_size = wsi.level_dimensions[level]
            while 0 in level_size:
                level -= 1
                level_size = wsi.level_dimensions[level]
            patch = wsi.read_region((0, 0), level, level_size)
            patch.close()
            correct_files.append(item)
        except openslide.OpenSlideUnsupportedFormatError:
            print svs_file, 'unsupported error' 
        except openslide.lowlevel.OpenSlideError:
            print svs_file, 'low level error'
    return correct_files
        
def _organize_and_divide_data():
    """The main rountine to generate the training patches.

    """
    logger = app_logger.get_logger('random_divide_data')
    # read the configuration file
    cfg_file = 'config/SJPath.cfg'
    config = ConfigParser.SafeConfigParser()
    
    logger.info('Using the config file: ' + cfg_file)
    config.read(cfg_file)
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    
    tumor_anno_files = glob.glob(os.path.join(setting.tumor_anno_folder, 
                                              '*.mask'))
    tumor_wsi_files = []
    normal_wsi_files = []
    for t in img_type:
        tumor_wsi_files.extend(glob.glob(os.path.join(setting.tumor_wsi_folder, t)))
        normal_wsi_files.extend(glob.glob(os.path.join(setting.normal_wsi_folder, t)))
    ########################################
    # normal_anno_files = glob.glob(os.path.join(setting.normal_anno_folder, 
    #                                            '*.mask'))
    # tumor_wsi_files = glob.glob(os.path.join(setting.tumor_wsi_folder, 
    #                                          '*.svs'))
    # normal_wsi_files = glob.glob(os.path.join(setting.normal_wsi_folder, 
    #                                           '*.svs'))
    print len(tumor_wsi_files), len(tumor_anno_files), len(normal_wsi_files)#, len(normal_anno_files)
    # normal_svs_to_masks = {}
    # for anno_file in normal_anno_files:
    #     svs_file = file_api.get_mask_info(os.path.basename(anno_file))[0]
    #     if svs_file not in normal_svs_to_masks:
    #         normal_svs_to_masks[svs_file] = [anno_file]
    #     else:
    #         normal_svs_to_masks[svs_file].append(anno_file)
            
    tumor_svs_to_masks = {}
    for anno_file in tumor_anno_files:
        svs_file = file_api.get_mask_info(os.path.basename(anno_file))[0]
        if svs_file not in tumor_svs_to_masks:
            tumor_svs_to_masks[svs_file] = [anno_file]
        else:
            tumor_svs_to_masks[svs_file].append(anno_file)
    print len(tumor_svs_to_masks)#, len(normal_svs_to_masks)
    
    # normal_svs_to_path = {}
    tumor_svs_to_path = {}
    # for file_path in normal_wsi_files:
    #     svs_file = os.path.basename(file_path).split('.')[0]
    #     normal_svs_to_path[svs_file] = file_path
    for file_path in tumor_wsi_files:
        svs_file = os.path.basename(file_path).split('.')[0]
        tumor_svs_to_path[svs_file] = file_path
    
    tumor_files = []
    normal_files = []
    # some annotation doesn't have corresponding WSIs 
    for svs_file, mask_files in tumor_svs_to_masks.iteritems():
        if svs_file in tumor_svs_to_path:
            tumor_files.append((tumor_svs_to_path[svs_file], mask_files))
    
    # for svs_file, mask_files in normal_svs_to_masks.iteritems():
	# if svs_file in normal_svs_to_path:
    #         normal_files.append((normal_svs_to_path[svs_file], mask_files))
    for svs_file in normal_wsi_files:
        normal_files.append((svs_file, None))

    tumor_files = _remove_corrupted_files(tumor_files)
    normal_files = _remove_corrupted_files(normal_files)
    
    logger.info('# of annotated normal WSIs: {}'.format(len(normal_files)))
    logger.info('# of annotated tumor WSIs: {}'.format(len(tumor_files)))
    logger.info('{} of the WSIs will be used for testing'
                .format(setting.test_frac))
    
    def change_split(item, name):
        item['split'] = name
        return item
    
    split_data = []
    train_files, test_files, val_files = _split_dataset(tumor_files, 
            setting.test_frac, setting.val_tumor)
    train_tumor_files = map(lambda item: change_split(item, 'Train_tumor'), train_files)
    val_tumor_files = map(lambda item: change_split(item, 'Val_tumor'), val_files)
    test_tumor_files = map(lambda item: change_split(item, 'Test_tumor'), test_files)
    
    split_data.extend(train_tumor_files)
    split_data.extend(val_tumor_files)
    split_data.extend(test_tumor_files)
    
    train_files, test_files, val_files = _split_dataset(normal_files, 
            setting.test_frac, setting.val_normal)
    train_normal_files = map(lambda item: change_split(item, 'Train_normal'), train_files)
    val_normal_files = map(lambda item: change_split(item, 'Val_normal'), val_files)
    test_normal_files = map(lambda item: change_split(item, 'Test_normal'), test_files)
    
    split_data.extend(train_normal_files)
    split_data.extend(val_normal_files)
    split_data.extend(test_normal_files)
    
    logger.info('# of training WSIs: {}'.format(len(train_tumor_files) 
                                + len(train_normal_files)))
    logger.info('# of validation WSIs: {}'.format(len(val_normal_files) 
                                + len(val_tumor_files)))
    logger.info('# of testing WSIs: {}'.format(len(test_normal_files) 
                                + len(test_tumor_files)))
    
    with open(setting.split_file, 'w') as f:
        json.dump(split_data, f)
        
    logger.info('All the val WSI file paths will be written to {}'
                .format(setting.test_file))
    with open(setting.test_file, 'w') as f:
        for item in test_tumor_files:
            f.write(item['WSI'][0] + '\n')
        for item in test_normal_files:
            f.write(item['WSI'][0] + '\n')

if __name__ == "__main__":
    _organize_and_divide_data()
