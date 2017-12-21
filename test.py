import glob
import os
import numpy as np
from skimage import measure
from scipy.ndimage import binary_opening
import ConfigParser

import app_logger

def classify_slides(prob_dir, class_result):      
    probmap_files = glob.glob(os.path.join(prob_dir, '*.npy'))
    with open(class_result, 'w') as f:
        for probmap_file in probmap_files:
            probmap = np.load(probmap_file)
            thresh_hold = probmap >= 0.5
            thresh_hold = binary_opening(thresh_hold, structure=np.ones((10, 10))).astype(np.int)
            
            prob = 0.0
            if probmap[thresh_hold != 0].sum() == 0:
                prob = 0
            else:
                prob = np.max(probmap[thresh_hold != 0])
            
            base = os.path.basename(probmap_file).split('.svs')[0]
            f.write(base + ' {}\n'.format(prob))
    
def detect_metastases(prob_dir, detect_result, downsample, receptive_field):      
    probmap_files = glob.glob(os.path.join(prob_dir, '*.npy'))
    for probmap_file in probmap_files:
        probmap = np.load(probmap_file)
        Xcorr = []
        Ycorr = []
        probs = []
        
        base = os.path.basename(probmap_file).split('.svs')[0]
        
        thresh_hold = probmap >= 0.5
        thresh_hold = binary_opening(thresh_hold, structure=np.ones((8, 8))).astype(np.int)
        
        if probmap[thresh_hold != 0].sum() != 0:
            thresh_hold = measure.label(thresh_hold, connectivity = 2) 
            properties = measure.regionprops(thresh_hold)
            
            for property in properties:
                if property.area > 100:
                    Xcorr.append(int((property.centroid[1] - 1) * downsample + receptive_field / 2))
                    Ycorr.append(int((property.centroid[0] - 1) * downsample + receptive_field / 2))
                    coords = property.coords
                    
                    prob = 0
                    for coord in coords:
                        if probmap[coord[0], coord[1]] > prob:
                            prob = probmap[coord[0], coord[1]]
                    probs.append(prob)
        
        if len(probs) != 0:
            save_file = os.path.join(detect_result, base + '.csv')
            with open(save_file, 'w') as f:
                for idx in range(len(probs)):
                    f.write('{} {} {}\n'.format(probs[idx], Xcorr[idx], Ycorr[idx]))
        
        

if __name__ == "__main__":
    cfg_file = 'config/SJPath.cfg'
    config = ConfigParser.SafeConfigParser()
    
    probmap_dir = config.get('TEST', 'probmap_dir')
    class_path = config.get('TEST', 'class_path')
    detect_dir = config.get('TEST', 'detect_dir')
    receptive_field = config.getint('TEST', 'receptive_field')
    downsample = config.getint('TEST', 'downsample')
    
    classify_slides(probmap_dir, class_path)
    detect_metastases(probmap_dir, detect_dir, downsample, receptive_field)