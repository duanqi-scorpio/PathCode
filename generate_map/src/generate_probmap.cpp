#include <iostream>
#include <fstream>
#include <ctime>
#include <opencv2/opencv.hpp>
#include <config4cpp/Configuration.h>
#include <boost/filesystem.hpp>

#include "WsiPredictor.h"
#include "openslide.h"
#include "cnpy.h"

using namespace cv;
using namespace config4cpp;
using namespace std;
using namespace boost::filesystem;

int main(int argc, char ** argv)
{
    /*
    const char        * config_file = "/home/bink//Code/SJTU_Path/VGG/CAMELYON_c/config/detect_config.cfg";
    WsiPredictor        predictor = WsiPredictor(config_file);  
    
    
    //const char *filename = "/media/usbdata/pathology/SJTU_PROJ/SJTU_Path/PathData/2017-05-16/4341.svs";
    const char *filename = "/media/usbdata/pathology/SJTU_PROJ/SJTU_Path/Normal/1/5418.svs";
    
    ifstream f(filename);
    CHECK(f.good()) << "File " << filename << " does not exist";
    openslide_t *p_wsi = openslide_open(filename);
    
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    predictor.predict(p_wsi);
    predictor.save_probmap("5418.npy", 0.3);
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    std::cout << elapsed << " seconds" << std::endl;
    openslide_close(p_wsi);
    */
    
    
    
    const char        * config_file = "../../config/detect_config.cfg";
    WsiPredictor        predictor = WsiPredictor(config_file); 
    
    const char *filename = "/media/usbdata/pathology/SJTU_PROJ/Experiments2/split/test_files.txt";
    std::string wsi_filename;
    std::ifstream file(filename);
    
    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);
    std::string savefile = "/media/usbdata/pathology/SJTU_PROJ/Experiments2/test/probmap/";
    while (std::getline(file, wsi_filename))
    {
        ifstream f(wsi_filename);
        CHECK(f.good()) << "File " << wsi_filename << " does not exist";
        openslide_t *p_wsi = openslide_open(wsi_filename.c_str());
        predictor.predict(p_wsi);
        string save_dir = savefile + wsi_filename.substr(wsi_filename.find(".svs") - 4) + ".npy";
        predictor.save_probmap(save_dir.c_str());
        //int level_count = openslide_get_level_count(p_wsi);
        //int64_t m_sz_wsi_height, m_sz_wsi_width;
        //int64_t width, height;
        //openslide_get_level_dimensions(p_wsi, 0, &m_sz_wsi_width, &m_sz_wsi_height);
        
        //for (int level = 1; level < level_count; ++level)
        //{
        //    openslide_get_level_dimensions(p_wsi, level, &width, &height);
        //    if (m_sz_wsi_width % width == 32)
        //    {
        //        break;
        //    }
        //}
        //cout << m_sz_wsi_height << '\t' << m_sz_wsi_width << endl;
        //cout << height << '\t' << width << endl;
        //cout << m_sz_wsi_height << '\t' << m_sz_wsi_width << '\t' << height << '\t' << width << endl;
        openslide_close(p_wsi);
    }
    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = (finish.tv_sec - start.tv_sec);
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    std::cout << elapsed << " seconds" << std::endl;
    
    return 0;
}

  