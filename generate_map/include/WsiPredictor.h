#ifndef WSI_PREDICTOR_H_
#define WSI_PREDICTOR_H_



//--------
// #include's
//--------
#include <string>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <map>

#include <config4cpp/Configuration.h>
#include <caffe/caffe.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "openslide.h"
#include "blockingconcurrentqueue.h"
#include "concurrentqueue.h"
#include "cnpy.h"
#include <iostream>

using namespace std;



//--------
// Class WsiPredictor
//
// Based on the trained Fully Convolutional Neural Network (FCN), 
// WSI_predictor generate a probability map for the whole slide image (WSI). 
// Each pixel in the prob. map indicates its prob. of belonging to tumor 
// regions.
//--------

struct kfbslide_t;
kfbslide_t* kfbslide_open(const char* filename);
void kfbslide_close(kfbslide_t* osr);

class WsiPredictor
{
private:
    struct QueueItem {
        QueueItem(){p_tile = NULL;}
        QueueItem(
                cv::Mat * p_img, 
                int tile_row, 
                int tile_col)
        {
            p_tile = p_img;
            row    = tile_row;
            col    = tile_col;
        }
        
        cv::Mat * p_tile;
        int       row;
        int       col;
    };
    
public:
    WsiPredictor(const char * cfg_file);
    ~WsiPredictor();
    
	void predict(openslide_t *  p_wsi);
    void predict(kfbslide_t* p_wsi);

    void save_probmap(
                std::string 	file_path,
                double      	sample_rate = 1);  

    void save_heatmap(
        std::string             file_path,
        double                  sample_rate = 1);
    
    void save_thumbimage(
        openslide_t*            p_wsi,
        std::string             file_path,
        double                  sample_rate = 1);

    void save_thumbimage(
        kfbslide_t*             p_wsi,
        std::string             file_path,
        double                  sample_rate = 1);
 
    
    static void _read_region_from_wsi(
                openslide_t *   p_wsi,
                cv::Mat &       result,                     
                const int64_t   x,
                const int64_t   y,
                const int32_t   level,
                const int64_t   w,
                const int64_t   h);

    static void kfb_read_region_from_wsi(
                kfbslide_t *    p_wsi,
                cv::Mat &       result,                     
                const int64_t   x,
                const int64_t   y,
                const int32_t   level,
                const int64_t   w,
                const int64_t   h);

private:
    std::string                 m_deploy_file;
    std::string                 m_model_file;
    int                         m_sz_downsample;
    int                         m_sz_recep_field;
    int                         m_sz_wd;
    int                         m_rate_ov;
    int                         m_lvl_ov;
    int                         m_off_set;
    int                         m_sz_tile_output;
    int                         m_num_row;
    int                         m_num_col;
    float                       m_factor_w;
    float                       m_factor_h;
    cv::Scalar                  m_mean_value;
    int64_t                     m_sz_wsi_width;
    int64_t                     m_sz_wsi_height;
    cv::Mat                     m_tile_LUT;
    cv::Mat                     m_probmap;
    std::vector<int>            m_gpu_ids;
    
    void _preprocessing(
                openslide_t *   p_wsi,
                cv::Mat &       result,
                const int64_t   width,
                const int64_t   height);
    void kfb_preprocessing(
                kfbslide_t *    p_wsi,
                cv::Mat &       result,
                const int64_t   width,
                const int64_t   height);
                
    void _calculate_tile_nums(openslide_t * p_wsi);
    void kfb_calculate_tile_nums(kfbslide_t *  p_wsi);

    void _tile_reader(
                openslide_t *                                       p_wsi,
                moodycamel::BlockingConcurrentQueue<QueueItem *> *  queue);
    void kfb_tile_reader(
                kfbslide_t *                                        p_wsi,
                moodycamel::BlockingConcurrentQueue<QueueItem *> *  queue);
                
    void _tile_processor(
                openslide_t *                                       p_wsi,
                const int                                           gpu_id,
                moodycamel::BlockingConcurrentQueue<QueueItem *> *  queue);
    void kfb_tile_processor(
                kfbslide_t *                                        p_wsi,
                const int                                           gpu_id,
                moodycamel::BlockingConcurrentQueue<QueueItem *> *  queue);
    
    inline void _get_tile_coordinates(
                const int       col,
                const int       row,
                int &           x_begin,
                int &           y_begin,
                int &           x_end,
                int &           y_end);
};


#endif  // WSI_PREDICTOR_H_
