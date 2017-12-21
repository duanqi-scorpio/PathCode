# Training and Testing code for SJTUPath

## Requirements for training

* tqdm
* Anaconda python 2.7
* openslide
* python lmdb
* pycaffe (exposed by environment variables)
* caffe tools (exposed by environment variables)

## Requirements for testing

* OpenCV
* Caffe
* libopenslide-dev


## Architecture

```
.
├── config                  # this directory contains configuration files
|   ├── logging.yaml        # python logging setting file
|   └── SJPath.cfg          # vqa setting file (directories, parameters, and so on)
├── test                    # my own testing code
|   └── ...
├── log                     # log dir containing the running logs
|   ├── errors.log          # error log file (automatically generated when the code is run)
|   ├── info.log            # running status file (automatically generated when the code is run)
|   ├── caffe-train.log     # training log for caffe model
|   └── ...
├── app_logger.py           # logging module
├── radom_divide_data.py    # randomly divide all the dataset into 3 splits
├── preprocess.py           # generate the training patches
├── single_WSI_processor.py # subroutine for preprocess.py
├── SJPath_helpers.py       # subroutine for preprocess.py
├── train.sh                # train caffe models
└── test.py                 # generate predictions
```



## Dataset Preparation
orgnize all the dataset like the following (see config/SJPath.cfg for more details)

```
.
└── sjpath_dir              # root folder of the whole dataset
    ├── normal_wsi_folder   # containing all the normal slides (normal_wsi_folder/*/*.svs)
    ├── normal_anno_folder  # containing all the annotations for normal slides (normal_wsi_folder/*.mask)
    ├── tumor_wsi_folder    # containing all the tumor slides (tumor_wsi_folder/*/*.svs)
    └── tumor_anno_folder   # containing all the annotations for tumor slides (tumor_wsi_folder/*.mask)
```

## Dataset Split
run radom_divide_data.py to randomly divide all the dataset into 3 split: training, validation and testing. 

* You can change how many slides you want to use for each split by setting test_frac, val_normal, val_tumor in DEFAULT section of config/SJPath.cfg file. 
* After running this code, all the training, validation and testing slides and their annotations all be stored in split_file in DEFAULT section of config/SJPath.cfg file in json format, which is required for preprocess.py. 

## Training Data Preparation
run preprocess.py to generate training, validation patches. After running this code, training patches (lmdb) are generated. Additionally, the pretrained model is downloaded and tranformed into FCN.

* By default, at most 1500 normal and tumor patches are generated from each slide. You can change this upper bound by changing sample_upper_bound in TRAIN section in config/SJTUPath.cfg. To see more intermediate results (annotated regions, tumor regions, selected trainin patches), you can go to directory cache_folder in the RESULT. 
* All the running infomation will be written into log/info.log. 
* Find the mean BGR values for training patches in the log. Find the transformed FCN model directory log.

## Training 
* change the GPUs you want to use for training in train.sh
* change the the directory of the pretrained model directory (transformed FCN model in the above section)
* change snapshot_prefix (line 15) in model/VGGSolver.prototxt. It's the directory you want to store the trained model
* Change BGR values in model/VGG_16_train_val.prototxt.
* Other changes should be obvious for a caffe user.
* run ./train.sh


## Generate probability map
* Install libopenslide-dev, Caffe, OpenCV
* If you build Caffe with cmake, comment the line 25 in generate_map/src/CMakeLists.txt and uncomment line 24
* Change filename (line 49) and savefile (line 56) in generate_probmap.cpp in generate_map/src/generate_probmap.cpp. filename stores the slides you want to generate probmap for. savefile is the directory you want to put probmaps
* Change model_file (line 4), mean_value (line 6), gpu_ids (line 8) in generate_map/config/detect_config.cfg. model_file is the trained model, mean_value is the BGR values used for training. gpu_ids is the GPU ids you want to use. You can use multiple GPUs. 
* cd to generate_map/build. run cmake .., run make all, run make install.
* cd to generate_map/install/bin. run ./generator to generate probmaps

## Testing
run test.py to generate classification and detection results
* numpy files in probmap_dir (section TEST of config/SJTUPath.cfg) will be used to generate the predictions
* classification result will be stored in class_path (section TEST of config/SJTUPath.cfg)
* detection result will be stored in detect_dir (section TEST of config/SJTUPath.cfg)