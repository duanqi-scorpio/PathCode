import ConfigParser
import json

import app_logger
import SJPath_helpers

def main():
    """The main rountine to generate the training patches.

    """
    logger = app_logger.get_logger('preprocess')
    
    # read the configuration file
    cfg_file = 'config/SJPath.cfg'
    config = ConfigParser.SafeConfigParser()
    
    logger.info('Using the config file: ' + cfg_file)
    config.read(cfg_file)
    generate_pkl = config.getboolean('RESULT', 'generate_pkl')
    save_train_patch = config.getboolean('RESULT', 'save_train_patch')
    train_patch_dir = config.get('RESULT', 'train_patch_dir')
        
    logger.info('Start to generate potential training tiles')
    if generate_pkl:
        SJPath_helpers.generate_label_dict_and_save_to_pkl(config)
    else:
        logger.info('label pkl file already generated')
    logger.info('Done generating potential training tiles')
    
    logger.info('Start saving patches to disk')
    if save_train_patch:
        SJPath_helpers.generate_training_patches(config)
        SJPath_helpers.save_training_patches_to_disk(config)
        logger.info('Finished saving patches to disk')
        
        logger.info('Start to convert training patches to lmdb')
        SJPath_helpers.convert_training_patches_to_lmdb(config)
    else:
        logger.info('Training patches are already saved to folder {}'
                    .format(train_patch_dir))
    
    logger.info('Computing image mean')
    SJPath_helpers.compute_img_mean(config)
    
    logger.info('Grabbing training patches and labels')
    SJPath_helpers.generate_train_txt(config)
    
    logger.info('Preparing pretrained model')
    SJPath_helpers.prepare_model(config)

if __name__ == "__main__":
    main()