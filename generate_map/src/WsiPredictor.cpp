#include "WsiPredictor.h"


//--------
// Inline functions
//--------

inline void 
WsiPredictor::_get_tile_coordinates(
    const int       col,
    const int       row,
    int &           x_begin,
    int &           y_begin,
    int &           x_end,
    int &           y_end)
{
    x_begin = (m_sz_wd - m_off_set) * col;
    y_begin = (m_sz_wd - m_off_set) * row;
    x_end   = x_begin + m_sz_wd;
    y_end   = y_begin + m_sz_wd;
         
    //--------
	// Check image boundary
	//-------- 
    x_end = (x_end <= m_sz_wsi_width ) ? x_end : m_sz_wsi_width;
    y_end = (y_end <= m_sz_wsi_height) ? y_end : m_sz_wsi_height;
}



//----------------------------------------------------------------------
// Function:	Constructor
//
// Description:	
//----------------------------------------------------------------------

WsiPredictor::WsiPredictor(const char * cfg_file)
{
	//--------
	// Open config file and read configurations to the member variables
	//--------    
    config4cpp::Configuration * cfg = config4cpp::Configuration::create();
    const char                * scope = "";
    config4cpp::StringVector    mean_value;
    config4cpp::StringVector    gpu_ids;

    try {
        cfg->parse(cfg_file);
        m_deploy_file       = cfg->lookupString(scope, "deploy_file");
        m_model_file        = cfg->lookupString(scope, "model_file");
        m_sz_wd             = cfg->lookupInt(scope, "window_size");
        m_rate_ov            = cfg->lookupInt(scope, "overview_rate");
        m_off_set           = cfg->lookupInt(scope, "offset");
        m_sz_tile_output    = cfg->lookupInt(scope, "tile_output_size");
        m_sz_downsample     = cfg->lookupInt(scope, "downsample_size");
        m_sz_recep_field    = cfg->lookupInt(scope, "receptive_field");
        
        cfg->lookupList(scope, "mean_value", mean_value);
        cfg->lookupList(scope, "gpu_ids", gpu_ids);
    } 
    catch(const config4cpp::ConfigurationException & ex) 
    {
        LOG(ERROR) << ex.c_str();
        //cfg->destroy();
    }
    cfg->destroy();
    
    //--------
	// Check if the receptive field related values are correctly given
	//--------  
    CHECK_EQ((m_sz_wd - m_sz_recep_field) % m_sz_downsample, 0) 
        << "window_size - receptive_field should be divisible by downsample_size";
        
    CHECK_EQ(m_sz_recep_field, m_off_set + m_sz_downsample) 
        << "offset need to be exactly receptive_field - downsample_size";

    CHECK_GT(m_rate_ov, 0)
        << "overview_rate need to be a positive value";
    
    //--------
	// Get image mean values and the gpu IDs
	//--------  
    std::vector<double> rgb_values;
    for (int idx = 0; idx < mean_value.length(); ++idx)
    {
        rgb_values.push_back(atof(mean_value[idx]));
    }
    CHECK_EQ(rgb_values.size(), 3) << "The mean values should be exactly 3";
    m_mean_value = cv::Scalar(rgb_values[0], rgb_values[1], rgb_values[2]);
    
    for (int idx = 0; idx < gpu_ids.length(); ++idx)
    {
        m_gpu_ids.push_back(atoi(gpu_ids[idx]));
    }
    CHECK_GT(m_gpu_ids.size(), 0) << "At least 1 GPU need to be given";
}



//----------------------------------------------------------------------
// Function:	Destructor
//
// Description:	
//----------------------------------------------------------------------

WsiPredictor::~WsiPredictor()
{
}



//----------------------------------------------------------------------
// Function:	predict
//
// Description:	
//----------------------------------------------------------------------

void
WsiPredictor::predict(openslide_t * p_wsi)
{
    //--------
	// Check the bad input pointer and get the size of the WSI
	//--------  
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
    
    openslide_get_level0_dimensions(p_wsi, &m_sz_wsi_width, &m_sz_wsi_height);
    CHECK(NULL == openslide_get_error(p_wsi))
        << "Get level 0 dimension error using openslide library.";    
    
    _calculate_tile_nums(p_wsi);
    
    //--------
	// Calculate the size of the prob. map and allocate memory for it
	//--------  
    int width  = m_num_col * m_sz_tile_output;
    int height = m_num_row * m_sz_tile_output;
    m_probmap  = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
    
    //--------
	// Start the single producer and multi consumer queue
	//--------  
    moodycamel::BlockingConcurrentQueue<QueueItem *> queue;
    std::thread tile_processors[m_gpu_ids.size()];
    std::thread tile_reader(&WsiPredictor::_tile_reader, 
                            this,
                            p_wsi,
                            &queue);
    
    for (size_t i = 0; i < m_gpu_ids.size(); ++i)
    {
        tile_processors[i] = std::thread(&WsiPredictor::_tile_processor, 
                                         this,
                                         p_wsi, 
                                         m_gpu_ids[i],
                                         &queue);
    }
    
    //--------
	// Wait for all the threads to stop
	//--------  
    tile_reader.join();
    for (size_t i = 0; i != m_gpu_ids.size(); ++i) 
    {
        tile_processors[i].join();
    }
}



//----------------------------------------------------------------------
// Function:	save_probmap
//
// Description:	Save the probmap into the specified file path
//----------------------------------------------------------------------

void 
WsiPredictor::save_probmap(
    std::string file_path,
    double      sample_rate)
{
    //--------
	// The actual size of the probmap
	//--------  
    int height = ceil((m_sz_wsi_height - m_sz_recep_field + m_sz_downsample) / 
                      float(m_sz_downsample));
    int width  = ceil((m_sz_wsi_width  - m_sz_recep_field + m_sz_downsample) / 
                      float(m_sz_downsample));
    LOG(INFO) << "The size of the heatmap: " << width << " * " << height;
                      
    //--------
	// Crop the probmap from the padded probmap
	//--------                    
    cv::Rect roi(0, 0, width, height);
    cv::Mat cropped_map = m_probmap(roi);
    
    height = ceil(height * sample_rate);
    width  = ceil(width  * sample_rate);
    
    cv::Mat output_map;
    cv::resize(cropped_map, output_map, cv::Size(width, height));
    
    //const unsigned int shape[] = {height, width};
    //cnpy::npy_save(file_path, (float*)output_map.data, shape, 2, "w");
    cnpy::npy_save(file_path, (float*)output_map.data, {height, width}, "w");
}


//----------------------------------------------------------------------
// Function:	_read_region_from_wsi
//
// Description:	Read a region from the WSI in a specified level and 
//   transform it into BGR channel order
//----------------------------------------------------------------------

void
WsiPredictor::_read_region_from_wsi(
    openslide_t *       p_wsi,
    cv::Mat &           result,                     
    const int64_t       x,
    const int64_t       y,
    const int32_t       level,
    const int64_t       w,
    const int64_t       h)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
    
    uint32_t * p_src = (uint32_t *)malloc(w * h * 4);
    CHECK(p_src) << "Unable to allocate enough memory";
    
    openslide_read_region(p_wsi, p_src, x, y, level, w, h);
    CHECK(NULL == openslide_get_error(p_wsi))
        << "Read region error with openslide library.";
    
    uchar * p_cur = (uchar *)p_src;
    for (int row = 0; row < h; ++row)
    {
        uchar * p_dest = result.ptr<uchar>(row);
        for (int col = 0; col < w; ++col)
        {
            uchar a = p_cur[3];
            uchar r = p_cur[2];
            uchar g = p_cur[1];
            uchar b = p_cur[0];
            
            if (a != 0 && a != 255) {
                r = r * 255 / a;
                g = g * 255 / a;
                b = b * 255 / a;
            }
            
            p_dest[0]   = b;
            p_dest[1]   = g;
            p_dest[2]   = r;
            p_dest      += 3;
            p_cur       += 4;
        }
    }
    
    free(p_src);
}	



//----------------------------------------------------------------------
// END OF PUBLIC API
//----------------------------------------------------------------------



//----------------------------------------------------------------------
// Function:	_preprocessing
//
// Description:	Get the oviewview level and threshold it
//----------------------------------------------------------------------

void 
WsiPredictor::_preprocessing(
    openslide_t *       p_wsi,
    cv::Mat &           result,
    const int64_t       width,
    const int64_t       height)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
        
    cv::Mat img_ov(height, width, CV_8UC3);
    _read_region_from_wsi( p_wsi, 
                           img_ov,
                           0,
                           0,
                           m_lvl_ov, 
                           width,
                           height);
    
    //--------
	// Convert the BGR channel order to HSV channel order
	//-------- 
    cv::Mat img_hsv;
    imwrite( "test.jpg", img_ov );
    cv::cvtColor(img_ov, img_hsv, CV_BGR2HSV);

    //--------
	// Grab the S channel and threshold it with OTSU method
	//--------     
    cv::Mat hsv_channels[3];
    cv::split(img_hsv, hsv_channels);
    
    result = hsv_channels[1];
    cv::GaussianBlur(result, result, cv::Size(9, 9), 0, 0);
    cv::threshold(result, result, 0, 255, cv::THRESH_OTSU); 
    
    cv::Mat element = getStructuringElement(cv::MORPH_CROSS, cv::Size(5, 5));
    cv::dilate(result, result, element, cv::Point(-1, -1));
    
    imwrite( "thresh.jpg", result );
}



//----------------------------------------------------------------------
// Function:	_calculate_tile_nums
//
// Description:	Calculate the number of tiles int the WSI, and record 
//    the info. in the member variables. Additionally, check all the 
//    tiles if they are in the tissue region.
//----------------------------------------------------------------------

void 
WsiPredictor::_calculate_tile_nums(
    openslide_t *   p_wsi)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
        
    //--------
	// Calculate the size of the overview level
	//-------- 
    int level_count = openslide_get_level_count(p_wsi);
    int64_t width, height;
    m_lvl_ov = level_count - 1;
    openslide_get_level_dimensions(p_wsi, m_lvl_ov, &width, &height);
    
    //cout << level_count << '\t' << m_lvl_ov << endl;
    m_factor_w = (float)m_sz_wsi_width / (float) width;
    m_factor_h = (float)m_sz_wsi_height / (float) height;
    
    //--------
	// Threashold the overview level 
	//-------- 
    cv::Mat img_thresh(height, width, CV_8UC1);
    _preprocessing(p_wsi, img_thresh, width, height);
    
    //--------
	// Calculate number of tiles in each dimension
	//-------- 
    m_num_row   = ceil((m_sz_wsi_height - m_sz_wd) / 
                       float(m_sz_wd - m_off_set)) + 1;
    m_num_col   = ceil((m_sz_wsi_width  - m_sz_wd) / 
                       float(m_sz_wd - m_off_set)) + 1;
    
    //--------
	// Check every tile if they are in the tissue region and record the info
    // in the opencv Mat object
	//--------           
    m_tile_LUT  = cv::Mat(m_num_row, m_num_col, CV_8UC1, cv::Scalar(0));
    
    for (int row = 0; row < m_num_row; ++row)
    {
        for (int col = 0; col < m_num_col; ++col)
        {
            int x_begin = 0;
            int y_begin = 0;
            int x_end   = 0;
            int y_end   = 0;
            _get_tile_coordinates(col, row, x_begin, y_begin, x_end, y_end);
            
            //--------
            // Transform the 0 level coords to the overview level
            //--------           
            int x_begin_ov = int(floor(x_begin / m_factor_w));
            int y_begin_ov = int(floor(y_begin / m_factor_h));
            int x_end_ov   = ceil(floor(x_end / m_factor_w));
            int y_end_ov   = ceil(floor(y_end / m_factor_h));
            
            //--------
            // Get the tile in the overview level and check if it is in the 
            // tissue region
            //--------  
            cv::Rect roi(x_begin_ov, 
                         y_begin_ov, 
                         x_end_ov - x_begin_ov, 
                         y_end_ov - y_begin_ov);
            cv::Mat tile_ov_th = img_thresh(roi);
            
            m_tile_LUT.at<uchar>(row, col) = 
                    (cv::sum(tile_ov_th)[0] == 0) ? 0 : 255;
        }
    }
}



//----------------------------------------------------------------------
// Function:	_tile_reader
//
// Description:	Read all the tiles in the tissue region of the WSI 
//   continuously into the queue
//----------------------------------------------------------------------

void
WsiPredictor::_tile_reader(
    openslide_t *                                       p_wsi,
    moodycamel::BlockingConcurrentQueue<QueueItem *> *  queue)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
        
    CHECK(NULL != queue)
        << "Please pass in a valid BlockingConcurrentQueue object pointer.";
        
    int height = m_tile_LUT.rows;
    int width  = m_tile_LUT.cols;
    
    for (int row = 0; row < height; ++row)
    {
        for (int col = 0; col < width; ++col)
        {
            if (0 != m_tile_LUT.at<uchar>(row, col))
            {
                int x_begin = 0;
                int y_begin = 0;
                int x_end   = 0;
                int y_end   = 0;
                _get_tile_coordinates(col, row, x_begin, y_begin, x_end, y_end);
                
                cv::Mat * tile = new cv::Mat(m_sz_wd, 
                                             m_sz_wd, 
                                             CV_8UC3, 
                                             cv::Scalar(0, 0, 0));
                
                //--------
                // If the tile lies in the border of the WSI, it need to padded
                //--------  
                if (y_end - y_begin < m_sz_wd || x_end - x_begin < m_sz_wd)
                {
                    cv::Mat temp_tile(y_end - y_begin, 
                                      x_end - x_begin, 
                                      CV_8UC3);
                    _read_region_from_wsi(p_wsi,
                                          temp_tile,
                                          x_begin,
                                          y_begin,
                                          0, 
                                          x_end - x_begin,
                                          y_end - y_begin);
                                          
                    //--------
                    // pad the tile
                    //--------  
                    cv::Mat roi = (*tile)(cv::Rect(0, 
                                                   0, 
                                                   x_end - x_begin,
                                                   y_end - y_begin));
                    temp_tile.copyTo(roi);
                }
                else
                {
                    _read_region_from_wsi(p_wsi,
                                          *tile,
                                          x_begin,
                                          y_begin,
                                          0, 
                                          m_sz_wd,
                                          m_sz_wd);
                }
                
                queue->enqueue(new QueueItem(tile, row, col));
            }
            
        }
    }
    
    //--------
    // It serves as sentinels to signal the tile processors to stop
    //--------  
    for (size_t gpu_id = 0; gpu_id < m_gpu_ids.size(); ++gpu_id)
    {
        queue->enqueue(new QueueItem());
    }
}



//----------------------------------------------------------------------
// Function:	_tile_processor
//
// Description:	Read all the tiles in the tissue region of the WSI 
//   continuously into the queue
//----------------------------------------------------------------------

void
WsiPredictor::_tile_processor(
    openslide_t *                                       p_wsi,
    const int                                           gpu_id,
    moodycamel::BlockingConcurrentQueue<QueueItem *> *  queue)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";

    CHECK(NULL != queue)
        << "Please pass in a valid BlockingConcurrentQueue object pointer.";
        
    //--------
	// Init caffe and set device
	//--------        
    caffe::Caffe::set_mode(caffe::Caffe::GPU);
    caffe::Caffe::SetDevice(gpu_id);
    
    //--------
	// Load caffe model
	//--------  
    caffe::Net<float> * net;
    net = new caffe::Net<float>(m_deploy_file, caffe::TEST);
    net->CopyTrainedLayersFrom(m_model_file);
    
    CHECK_EQ(net->num_inputs(), 1) 
            << "Network should have exactly one input.";
    CHECK_EQ(net->num_outputs(), 1) 
            << "Network should have exactly one output.";
    
    //--------
	// Check the channel numbers
	//--------     
    caffe::Blob<float>* input_layer = net->input_blobs()[0];
    int num_channels                = input_layer->channels();
    CHECK(num_channels == 3)        << "Input layer should have 3 channels.";

    //--------
	// Reshape the input layer
	//--------         
    input_layer->Reshape(1, 3, m_sz_wd, m_sz_wd);
    net->Reshape();
    
            
    //--------
	// Process the tiles until all the queue is empty 
	//--------      
    bool is_wsi_region = true;
    while (is_wsi_region)
    {
        QueueItem * item;
        queue->wait_dequeue(item);
        cv::Mat * tile = item->p_tile;
        int row        = item->row;
        int col        = item->col;
        delete item;
        item = NULL;
        
        if (NULL != tile)
        {
            //--------
            // Wrap the input layer in a Mat object to facilitate copying
            //--------     
            std::vector<cv::Mat> input_channels;
            float* input_data = input_layer->mutable_cpu_data();
            for (int i = 0; i < num_channels; ++i) 
            {
                cv::Mat channel(m_sz_wd, m_sz_wd, CV_32FC1, input_data);
                input_channels.push_back(channel);
                input_data += m_sz_wd * m_sz_wd;
            }
    
            //--------
            // Preprocess the input image
            //--------  
            cv::Mat sample_float;
            tile->convertTo(sample_float, CV_32FC3);
            sample_float -= m_mean_value;
            
            delete tile;
            tile = NULL;
            
            cv::split(sample_float, input_channels);
            CHECK(reinterpret_cast<float*>(input_channels.at(0).data)
                  == net->input_blobs()[0]->cpu_data())
                << "Input channels are not wrapping the input layer of the network.";
                
            net->Forward();
            
            //--------
            // Wrap the output layer in a Mat object to facilitate copying
            //--------       
            caffe::Blob<float>* output_layer = net->output_blobs()[0];
            int width                        = output_layer->width();
            int height                       = output_layer->height();
            float* output_data               = output_layer->mutable_cpu_data() + 
                                               width * height;
            cv::Mat output(height, width, CV_32FC1, output_data);
            //cout << cv::sum(output)[0] << endl;
            
            //--------
            // Copy the output to the probability map
            //--------  
            int col_begin = col * m_sz_tile_output;
            int row_begin = row * m_sz_tile_output;
            
            cv::Mat roi = m_probmap(cv::Rect(col_begin, 
                                             row_begin, 
                                             m_sz_tile_output,
                                             m_sz_tile_output));
            //cout << roi.rows << '\t' << roi.cols << '\t' << output.rows << '\t' << output.cols << endl;       
            output.copyTo(roi);
        }
        else
        {
            is_wsi_region = false;
        }
    }
    delete net;
    net = NULL;
}
