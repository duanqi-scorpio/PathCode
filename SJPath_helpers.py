import numpy as np
from tqdm import tqdm
import json
import os
import sys
import pickle
from random import shuffle
from PIL import Image
import glob
import urllib

import openslide
import file_api
import lmdb
sys.path.append("/home/fengyifan/caffe/python/")

import caffe

import app_logger
import single_WSI_processor

class SJPathSetting(object):
    def __init__(self):
        pass
        
    def parse(self, config):
        self.split_file = config.get('DEFAULT', 'split_file')
        self.patch_size = config.getint('TRAIN', 'patch_size')
        self.patch_factor = config.getint('TRAIN', 'patch_factor')
        
        # cache folder
        self.img_vis = config.get('RESULT', 'img_vis')
        self.img_vis_patch = config.get('RESULT', 'img_vis_patch')
        self.img_vis_patch_prob = config.getfloat('RESULT', 'img_vis_patch_prob')
        self.img_ext = config.get('DEFAULT', 'img_ext')
        self.downsample_rate = config.getfloat('RESULT', 'downsample_rate')
        self.save_img = config.getboolean('RESULT', 'save_img')
        self.train_patch_dir = config.get('RESULT', 'train_patch_dir')
        
        # do we need to generate label pkl file
        self.pkl_folder = config.get('RESULT', 'pkl_folder')
        
        self.sample_upper_bound = config.getint('TRAIN', 'sample_upper_bound')
        
        # the final training files
        self.db_folder = config.get('TRAIN', 'db_folder')
        self.db_batch = config.getint('TRAIN', 'db_batch')
        
        # the directory to store label txt files
        self.train_txt = config.get('TRAIN', 'train_txt')
        self.val_txt = config.get('TRAIN', 'val_txt')
        
        # pretrained model
        self.pretrained_url = config.get('TRAIN', 'pretrained_url')
        self.pretrained_dir = config.get('TRAIN', 'pretrained_dir')
        self.pretrained_model = config.get('TRAIN', 'pretrained_model')
        self.original_prototxt = config.get('TRAIN', 'original_prototxt')
        self.full_conv_prototxt = config.get('TRAIN', 'full_conv_prototxt')
        self.full_conv_model = config.get('TRAIN', 'full_conv_model')
        self.trained_dir = config.get('TRAIN', 'trained_dir')
        
        # predictions
        self.probmap_dir = config.get('TEST', 'probmap_dir')
        self.class_dir = config.get('TEST', 'class_dir')
        self.detect_dir = config.get('TEST', 'detect_dir')
        
        if not os.path.exists(self.img_vis):
            os.makedirs(self.img_vis)
            
        if not os.path.exists(self.img_vis_patch):
            os.makedirs(self.img_vis_patch)

        if not os.path.exists(self.trained_dir):
            os.makedirs(self.trained_dir)
            
        if not os.path.exists(self.pretrained_dir):
            os.makedirs(self.pretrained_dir)
            
        if not os.path.exists(self.pkl_folder):
            os.makedirs(self.pkl_folder)
            
        if not os.path.exists(self.train_patch_dir):
            os.makedirs(self.train_patch_dir)
            os.makedirs(os.path.join(self.train_patch_dir, 'TRAIN/pos'))
            os.makedirs(os.path.join(self.train_patch_dir, 'TRAIN/neg'))
            os.makedirs(os.path.join(self.train_patch_dir, 'VAL/pos'))
            os.makedirs(os.path.join(self.train_patch_dir, 'VAL/neg'))
            
        if not os.path.exists(self.db_folder):
            os.makedirs(self.db_folder)
            
        if not os.path.exists(self.probmap_dir):
            os.makedirs(self.probmap_dir)
        if not os.path.exists(self.class_dir):
            os.makedirs(self.class_dir)
        if not os.path.exists(self.detect_dir):
            os.makedirs(self.detect_dir)

def _prepare_all_patch(split_data, setting):
    logger = app_logger.get_logger('preprocess')
    processor = single_WSI_processor.WSIProcessor(setting)
    
    logger.info('Processing Training WSIs')
    preproces_data = filter(lambda item: item['split'] != 'Test_tumor' and 
                                     item['split'] != 'Test_normal', split_data)
    
    all_train_tiles = []
    for idx, item in enumerate(tqdm(preproces_data)):
        label_dict = None
        if 'tumor' in item['split']:
            label_dict = processor.process(item, 'pos')
            label_dict['neg']=[]
        if 'normal' in item['split']:
            label_dict = processor.process(item, 'neg')
            label_dict['pos']=[]
        count_pos = len(label_dict['pos'])
        count_neg = len(label_dict['neg'])
        all_train_tiles.append({'WSI': item['WSI'][0], 
                                'split': item['split'], 'tiles': label_dict})
        logger.info('Initial patch # for {}, # of pos tile: {}, # of neg tile: {}'
                    .format(os.path.basename(item['WSI'][0]), count_pos, count_neg))
                    
    return all_train_tiles
    """
    label_dict = {}
    for idx, wsi_gt_pair in enumerate(tqdm(all_files)):
        label_dict[wsi_gt_pair['name']] = processor.process(wsi_gt_pair)
    
    if not os.path.exists(setting.result_file_root):
        os.makedirs(setting.result_file_root)
    
    # save label dict to pickle file
    pickle_file = os.path.join(setting.result_file_root, 'label_dict.pkl')
    logger.info('Label dict will be saved into: {}'. format(pickle_file))
    with open(pickle_file, 'wb') as f:
        pickle.dump(label_dict, f)
    """
            

def generate_label_dict_and_save_to_pkl(
    config
):
    logger = app_logger.get_logger('preprocess')
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    
    with open(setting.split_file) as f:    
        split_data = json.load(f)
        
    train_data = filter(lambda item: item['split'] == 'Train_tumor' or 
                                     item['split'] == 'Train_normal', split_data)
    val_data = filter(lambda item: item['split'] == 'Val_tumor' or 
                                   item['split'] == 'Val_normal', split_data)
    test_data = filter(lambda item: item['split'] == 'Test_tumor' or 
                                    item['split'] == 'Test_normal', split_data)
                                    
    logger.info('# of training WSIs: {}'.format(len(train_data)))
    logger.info('# of validation WSIs: {}'.format(len(val_data)))
    logger.info('# of testing WSIs: {}'.format(len(test_data)))
    all_labels = _prepare_all_patch(split_data, setting)
    
    pickle_file = os.path.join(setting.pkl_folder, 'label_dict.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(all_labels, f) 
    logger.info('label dict file is saved to {}'.format(pickle_file))

def _random_sample_tiles(all_tiles, sample_upper_bound):
    sampled_tiles = []
    for item in all_tiles:
        svs_file = item['WSI']
        label_tiles = item['tiles']
        
        label_dict = {}
        label_dict['WSI'] = svs_file
        label_dict['split'] = item['split']
        label_dict['tiles'] = {}
        for key in label_tiles.keys():
            key_tiles = label_tiles[key]
            curr_count = len(key_tiles)
            count_new = min(sample_upper_bound, curr_count)
            
            indices = range(curr_count)
            shuffle(indices)
            new_list = [None] * count_new
            
            for idx in range(count_new):
                new_list[idx] = key_tiles[indices[idx]]
            label_dict['tiles'][key] = new_list
        sampled_tiles.append(label_dict)
    return sampled_tiles
    
def generate_training_patches(
    config
):
    logger = app_logger.get_logger('preprocess')
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    pickle_file = os.path.join(setting.pkl_folder, 'label_dict.pkl')
    if not os.path.exists(pickle_file):
        logger.error('{} does not exists, please regenerate the file'.format(pickle_file))
        
    logger.info('Load label dict file drom {}'.format(pickle_file))
    with open(pickle_file, 'rb') as f:
        all_tiles = pickle.load(f)
    logger.info('Tiles for {} of WSI are loaded'.format(len(all_tiles)))
    
    logger.info('Randomly sampling training tiles')
    sampled_tiles = _random_sample_tiles(all_tiles, setting.sample_upper_bound)
    logger.info('Done sampling training tiles')
    
    for item in sampled_tiles:
        svs_file = item['WSI']
        label_dict = item['tiles']
        logger.info('Number of tiles sampled for {}, # of pos: {}, # of neg: {}'
                    .format(os.path.basename(svs_file), len(label_dict['pos']), len(label_dict['neg'])))
        
        # visualize the sampled training patches in the mask file
        if setting.save_img:
            mask_file = os.path.join(setting.img_vis, 
                                     os.path.basename(svs_file)[:-4] + '_mask' + setting.img_ext)
            lvl_mask = Image.open(mask_file)
            lvl_mask = np.asarray(lvl_mask)
            lvl_mask.flags.writeable = True
            
            for key, label_tiles in label_dict.iteritems():
                for (col, row) in label_tiles:
                    col = col / setting.patch_factor
                    row = row / setting.patch_factor
                    
                    # label selected tumor as yellow and normal as white
                    if key == 'pos':
                        lvl_mask[row, col, :] = [255, 255, 0]
                    else:
                        lvl_mask[row, col, :] = [255, 255, 255]
            lvl_mask = Image.fromarray(lvl_mask)
            lvl_mask.save(os.path.join(setting.img_vis, 
                               os.path.basename(svs_file)[:-4] + '_mask_selected' + setting.img_ext))
                               
    pickle_file = os.path.join(setting.pkl_folder, 'sampled_label_dict.pkl')
    with open(pickle_file, 'wb') as f:
        pickle.dump(sampled_tiles, f) 
    logger.info('Sampled label dict file is saved to {}'.format(pickle_file))

    
def save_training_patches_to_disk(
    config
):
    logger = app_logger.get_logger('preprocess')
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    pickle_file = os.path.join(setting.pkl_folder, 'sampled_label_dict.pkl')
    if not os.path.exists(pickle_file):
        logger.error('{} does not exists, please regenerate the file'.format(pickle_file))
        
    logger.info('Load sampled label dict file drom {}'.format(pickle_file))
    with open(pickle_file, 'rb') as f:
        sampled_tiles = pickle.load(f)
    logger.info('Sampled tiles for {} of WSIs are loaded'.format(len(sampled_tiles)))
    
    logger.info('Saving sampled training tiles to disk')
    for item in tqdm(sampled_tiles):
        svs_file = item['WSI']
        wsi = file_api.AllSlide(svs_file)
        label_dict = item['tiles']
        split = item['split']
        base = setting.train_patch_dir
        
        if split in ['Train_tumor', 'Train_normal']:
            base = os.path.join(base, 'TRAIN')
        else:
            base = os.path.join(base, 'VAL')
        
        for key, label_tiles in label_dict.iteritems():
            base_label = os.path.join(base, key)
            base_label = os.path.join(base_label, os.path.basename(svs_file)[: -4])
            for origin in label_tiles:
                patch = wsi.read_region(origin, 0, 
                                        (setting.patch_size, setting.patch_size))
                patch = patch.convert('RGB')
                patch.save(base_label + '_{}_{}.tif'.format(origin[0], origin[1]))
                patch.close()
                
def convert_training_patches_to_lmdb(
    config
):
    logger = app_logger.get_logger('preprocess')
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    
    # find all the training and validation patches on the disk
    base = setting.train_patch_dir
    logger.info('Grabbing image files from {}'.format(base))
    train_pos_patches = glob.glob(os.path.join(base, 'TRAIN/pos/*.tif'))
    train_neg_patches = glob.glob(os.path.join(base, 'TRAIN/neg/*.tif'))
    val_pos_patches = glob.glob(os.path.join(base, 'VAL/pos/*.tif'))
    val_neg_patches = glob.glob(os.path.join(base, 'VAL/neg/*.tif'))
    
    # blend samples
    train_patches = []
    val_patches = []
    train_patches.extend(map(lambda x: {'filename':x, 'label': 1}, 
                             train_pos_patches))
    train_patches.extend(map(lambda x: {'filename':x, 'label': 0}, 
                             train_neg_patches))
    val_patches.extend(map(lambda x: {'filename':x, 'label': 1}, 
                           val_pos_patches))
    val_patches.extend(map(lambda x: {'filename':x, 'label': 0}, 
                             val_neg_patches)) 
                             
    logger.info('# of training patches: {} (# of pos: {}, # of neg: {})'
                .format(len(train_patches), len(train_pos_patches), 
                        len(train_neg_patches)))
    logger.info('# of validation patches: {} (# of pos: {}, # of neg: {})'
                .format(len(val_patches), len(val_pos_patches), 
                        len(val_neg_patches)))
    
    # randomrize training
    shuffle(train_patches)
    shuffle(train_patches)
    shuffle(val_patches)
    shuffle(val_patches)
    def _write_to_db(patches, file_dir, db_batch):
        batch_imgs = []
        
        # create the lmdb file
        lmdb_env = lmdb.open(file_dir, map_size=int(1e12))
        lmdb_txn = lmdb_env.begin(write=True)
        datum = caffe.proto.caffe_pb2.Datum()
        
        batch_idx = 0
        for idx, item in enumerate(tqdm(patches)):
            patch = Image.open(item['filename'])
            img = np.asarray(patch)
            label = item['label']
            patch.close()
            
            # map numpy array order to caffe order
            img = img[:, :, (2, 1, 0)]  # RGB to BGR
            img = img.transpose((2, 0, 1))  # cxhxw in Caffe
            
            # save in datum
            datum = caffe.io.array_to_datum(img, label)
            keystr = '{:0>8d}'.format(idx)
            lmdb_txn.put(keystr, datum.SerializeToString())
            
            if idx % db_batch == 0:
                #print idx, batch_idx
                lmdb_txn.commit()
                lmdb_txn = lmdb_env.begin(write=True)
        if idx % db_batch != 0:
            lmdb_txn.commit()
#    def _write_to_db(patches, file_dir, db_batch):
#        batch_imgs = []
#        def _write_batch(batch_imgs, db_dir, batch_idx):
#            db_saver = leveldb.LevelDB(db_dir)
#            batch = leveldb.WriteBatch()
#            for idx, im_label in enumerate(batch_imgs):
#                im_dat = caffe.io.array_to_datum(im_label[0])   # image content
#                im_dat.label = label    # label content
#                batch.Put('{:0>10d}'.format(batch_idx) + '{:0>10d}'.format(idx),
#                          im_dat.SerializeToString())
#            db_saver.Write(batch, sync=True)
#        
#        batch_idx = 0
#        for idx, item in enumerate(tqdm(patches)):
#            patch = Image.open(item['filename'])
#            img = np.asarray(patch)
#            label = item['label']
#            patch.close()
#            
#            # map numpy array order to caffe order
#            img = img[:, :, (2, 1, 0)]  # RGB to BGR
#            img = img.transpose((2, 0, 1))  # cxhxw in Caffe
#            batch_imgs.append([img, label])
#            
#            if idx % db_batch == 0 or idx == len(patches) - 1:
#                #print idx, batch_idx
#                _write_batch(batch_imgs, file_dir, batch_idx)
#                batch_imgs = []
#                batch_idx += 1
            
    # write to leveldb
    file_dir = os.path.join(setting.db_folder, 'TRAIN_lmdb')
    logger.info('Writing training patches to lmdb {}'.format(file_dir))
    _write_to_db(train_patches, file_dir, setting.db_batch)
    
    file_dir = os.path.join(setting.db_folder, 'VAL')
    logger.info('Writing validation patches to lmdb {}'.format(file_dir))
    _write_to_db(val_patches, file_dir, setting.db_batch)

def compute_img_mean(
    config
):
    logger = app_logger.get_logger('preprocess')
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    
    db_dir = os.path.join(setting.db_folder, 'TRAIN_lmdb')
    mean_file = os.path.join(setting.db_folder, 'mean.binaryproto')
    
    mean_bin = 'compute_image_mean'
    cmd_line = mean_bin + ' ' + db_dir + ' ' + mean_file + ' ' + '-backend ' + 'lmdb'
    logger.info(cmd_line)
    
    # calculating mean image file
    try:
        os.system(cmd_line)
    except IOError as e:
        logger.error("Compute_image_mean I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError as e:
        logger.error("Compute_image_mean ValueError error({0}): {1}".format(e.errno, e.strerror))
    except:
        logger.error('Compute_image_mean unexpected error')
    
    # calculate mean values
    blob = caffe.proto.caffe_pb2.BlobProto()
    try:
        mean_values = np.mean(out, axis=(1, 2)).tolist()
        str_means = [str(k) for k in mean_values]
        str_means = ';'.join(str_means)
        logger.info('mean values: ' + str_means)
        
        data = open(mean_file, 'rb').read()
        blob.ParseFromString(data)
        out = np.array(caffe.io.blobproto_to_array(blob))[0]
        #print out.shape
    except IOError as e:
        logger.error("computeMeanValueFromFile I/O error({0}): {1}".format(e.errno, e.strerror))
    except ValueError as e:
        logger.error("computeMeanValueFromFile ValueError error({0}): {1}".format(e.errno, e.strerror))
    except:
        logger.error('computeMeanValueFromFile unexpected error')
    return None
    
def generate_train_txt(
    config
):
    logger = app_logger.get_logger('preprocess')
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    
    # find all the training and validation patches on the disk
    base = setting.train_patch_dir
    logger.info('Grabbing image files from {}'.format(base))
    train_pos_patches = glob.glob(os.path.join(base, 'TRAIN/pos/*.tif'))
    train_neg_patches = glob.glob(os.path.join(base, 'TRAIN/neg/*.tif'))
    val_pos_patches = glob.glob(os.path.join(base, 'VAL/pos/*.tif'))
    val_neg_patches = glob.glob(os.path.join(base, 'VAL/neg/*.tif'))

    train_pos_files = map(lambda x: (x, '1'), train_pos_patches)
    train_neg_files = map(lambda x: (x, '0'), train_neg_patches)
    val_pos_files = map(lambda x: (x, '1'), val_pos_patches)
    val_neg_files = map(lambda x: (x, '0'), val_neg_patches)

    train_files = []
    train_files.extend(train_pos_files)
    train_files.extend(train_neg_files)

    val_files = []
    val_files.extend(val_pos_files)
    val_files.extend(val_neg_files)

    shuffle(train_files)
    shuffle(train_files)

    shuffle(val_files)
    shuffle(val_files)
    #print len(train_files), len(val_files)

    #train_txt = '/media/usbdata/pathology/SJTU_PROJ/Experiments/train/train.txt'
    #val_txt = '/media/usbdata/pathology/SJTU_PROJ/Experiments/train/val.txt'

    logger.info('train patches and their labels will be written into {}'
                .format(setting.train_txt))
    with open(setting.train_txt, 'w') as f:
        for item in train_files:
            f.write(item[0] + ' ' + item[1] + '\n')
        
    logger.info('validation patches and their labels will be written into {}'
                .format(setting.val_txt))
    with open(setting.val_txt, 'w') as f:
        for item in val_files:
            f.write(item[0] + ' ' + item[1] + '\n')
 
def prepare_model(
    config
):
    logger = app_logger.get_logger('preprocess')
    
    # get the basic dataset and setting info
    setting = SJPathSetting()
    setting.parse(config)
    
    logger.info('Downloading pretrained model, this may take a while')
    #urllib.urlretrieve(setting.pretrained_url, 
    #                   filename=setting.pretrained_model)
                       
    global rem_file # global variable to be used in dlProgress
    rem_file = setting.pretrained_url
    def dlProgress(count, blockSize, totalSize):
        percent = int(count*blockSize*100/totalSize)
        sys.stdout.write("\r" + rem_file + "...%d%%" % percent)
        sys.stdout.flush()

    if os.path.exists(setting.pretrained_model):
        print('find model!')
    else:
        urllib.urlretrieve(rem_file, filename=setting.pretrained_model,
                           reporthook=dlProgress)
    
    net = caffe.Net(setting.original_prototxt, setting.pretrained_model, 
                    caffe.TEST)
    net_full_conv = caffe.Net(setting.full_conv_prototxt, 
            setting.pretrained_model, caffe.TEST)

    params = ['fc6', 'fc7']
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) 
                 for pr in params}

    params_full_conv = ['fc6_conv', 'fc7_conv']
    conv_params = {pr: (net_full_conv.params[pr][0].data,
                        net_full_conv.params[pr][1].data) 
                   for pr in params_full_conv}

    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]
    
    logger.info('The model is transformed to FCN, it will be written to {}'
                .format(setting.full_conv_model))
    net_full_conv.save(setting.full_conv_model)
 
if __name__ == "__main__":
    pass
