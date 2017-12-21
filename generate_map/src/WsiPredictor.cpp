#include "WsiPredictor.h"

#define OPENSLIDE_PROPERTY_NAME_MPP_X "openslide.mpp-x"
#define OPENSLIDE_PROPERTY_NAME_MPP_Y "openslide.mpp-t"
#define OPENSLIDE_PROPERTY_NAME_VENDOR "openslide.vendor"

typedef struct IMAGE_INFO_STRUCT
{
    int DataFilePTR;  // Linux version
    // LPARAM DataFilePTR; // Windows version
} ImageInfoStruct;

struct dimension_t {
    int64_t  h;
    int64_t  w;
    double   downsample;
};

struct AssociateImage
{
    int   m_iWidth;
    int   m_iHeight;
    int   m_iDataLength;
    unsigned char*  m_ImagePixel;
};

struct kfbslide_t
{
    ImageInfoStruct  m_imgInfo;
    int64_t          m_iWidth;
    int64_t          m_iHeight;
    float            m_fScale;
    float            m_fSpendTime;
    double           m_dScanTime;
    float            m_fImgCapRes;
    int              m_iBlockSize;
    int              m_iLevelCount;
    dimension_t*     m_pDimensions;
    std::string      m_sFileName;

    const char**  m_pPropertiesNames;
    std::map<std::string, std::string >  m_mapProperties;

    std::map< std::string, AssociateImage>    m_mapAssociateImgs;
    const char**   m_pAssociateImageNames;
};

namespace {
	std::string get_kfb_lib_file_path()
	{
		return "libImageOperationLib.so";
	}
}

typedef void* LPVOID;
const int MaxKFBLevel = 5;
#ifndef BYTE
#define BYTE unsigned char
#endif

#define CHECK_NULL(ptr)  \
      if(!ptr) \
         return false;

class KfbLib
{
protected:
	typedef int (*InitImageFileFunc)(ImageInfoStruct* sImageInfo, const char* Path);

	typedef int (*UnInitImageFileFunc)(ImageInfoStruct* sImageInfo);

	typedef int (*DeleteImageDataFunc)(LPVOID pImageData);

    typedef int (*GetHeaderInfoFunc)(ImageInfoStruct* sImageInfo, int* khiImageHeight,
        int* khiImageWidth, int* khiScanScale, float* khiSpendTime, double* khiScanTime,
        float* khiImageCapRes, int* khiImageBlockSize);

    typedef char* (*GetImageStreamFunc)(ImageInfoStruct* sImageInfo, float fScale,
        int nImagePosX, int nImagePosY, int* nDataLength, unsigned char** ImageStream);

    typedef char* (*GetImageRGBDataStreamFunc)(ImageInfoStruct* sImageInfo, float fScale,
        int nImagePosX, int nImagePosY, int* nDataLength, int * nImageWidth,
        int * nImageHeight,unsigned char** ImageStream);

	typedef int (*GetImageDataRoiFunc)(ImageInfoStruct* sImageInfo, float fScale,
		int sp_x, int sp_y, int nWidth, int nHeight, BYTE** pBuffer, int *DataLength, bool flag);

    typedef int (*GetThumnailImagePathFunc)(const char* szFilePath, unsigned char** ImageData,
        int* nDataLength, int* nThumWidth, int* nThumHeght);
    typedef int (*GetPriviewInfoPathFunc)(const char* szFilePath, unsigned char** ImageData,
        int* nDataLength, int* nThumWidth, int* nThumHeght);
	typedef int (*GetLableInfoPathFunc)(const char* szFilePath, unsigned char** ImageData,
        int* nDataLength, int* nThumWidth, int* nThumHeght);

	typedef int (*GetAssociateImageFunc)(const char* szFilePath, unsigned char** ImageData,
        int* nDataLength, int* nThumWidth, int* nThumHeght);

protected:
	InitImageFileFunc           m_initImage;
	UnInitImageFileFunc         m_UnInitImage;
	DeleteImageDataFunc         m_DeleteImageDataFunc;
	GetHeaderInfoFunc           m_GetHeaderInfo;
	GetImageStreamFunc          m_GetImageStream;
	GetImageDataRoiFunc         m_GetImageDataRoiFunc;
    GetImageRGBDataStreamFunc   m_GetImageRGBDataStreamFunc;
	GetThumnailImagePathFunc    m_GetThumbImg;
	GetPriviewInfoPathFunc      m_GetPreviewImg;
	GetLableInfoPathFunc        m_GetLabelImage;

public:
    static KfbLib* Instance()
    {
        if ( !m_kfbLib )
        {
            m_kfbLib = new KfbLib( get_kfb_lib_file_path() );
            if ( !m_kfbLib->Initialize() )
            {
                printf("Fail to Initialize KfbLib\n");
                return NULL;
            }
        }
        return m_kfbLib;
    }

    ~KfbLib()
    {
        if ( m_pKfbLib )
        {
            printf("Destruct m_pKfbLib\n");
            dlclose(m_pKfbLib);         // Linux
			// FreeLibrary(m_pKfbLib);  // Windows
        }
    }

public:
    bool Initialize()
    {
        // printf("Lib file Path : %s\n", m_sLibFile.c_str());
        // m_pKfbLib = LoadLibrary((LPCSTR)(m_sLibFile.c_str()));       // Windows
        m_pCurlLib = dlopen("libcurl.so", RTLD_NOW);
        m_pJpegLib = dlopen("libjpeg.so.9", RTLD_NOW);
        m_pKfbLib  = dlopen("libImageOperationLib.so", RTLD_NOW);              //Linux
        if ( !m_pKfbLib )
        {
            char* pErr = dlerror();
            printf("Fail to load kfb library, Error is: %s\n", pErr);
            return false;
        }
        dlerror();
        m_initImage = (InitImageFileFunc)dlsym(m_pKfbLib, "InitImageFileFunc");
		// m_initImage = (InitImageFileFunc)GetProcAddress(m_pKfbLib, "InitImageFileFunc");
        CHECK_NULL( m_initImage );

        m_UnInitImage = (UnInitImageFileFunc)dlsym(m_pKfbLib, "UnInitImageFileFunc");
		// m_UnInitImage = (UnInitImageFileFunc)GetProcAddress(m_pKfbLib, "UnInitImageFileFunc");
        CHECK_NULL( m_UnInitImage );

        m_GetImageStream = (GetImageStreamFunc)dlsym(m_pKfbLib, "GetImageStreamFunc");
		// m_GetImageStream = (GetImageStreamFunc)GetProcAddress(m_pKfbLib, "GetImageStreamFunc");
        CHECK_NULL( m_GetImageStream );

        m_GetHeaderInfo = (GetHeaderInfoFunc)dlsym(m_pKfbLib, "GetHeaderInfoFunc");
		// m_GetHeaderInfo = (GetHeaderInfoFunc)GetProcAddress(m_pKfbLib, "GetHeaderInfoFunc");
        CHECK_NULL(m_GetHeaderInfo);

        m_GetThumbImg = (GetThumnailImagePathFunc)dlsym(m_pKfbLib, "GetThumnailImagePathFunc");
		// m_GetThumbImg = (GetAssociateImageFunc)GetProcAddress(m_pKfbLib, "GetThumnailImagePathFunc");
        CHECK_NULL( m_GetThumbImg );

        m_GetPreviewImg = (GetPriviewInfoPathFunc)dlsym(m_pKfbLib, "GetPriviewInfoPathFunc");
		// m_GetPreviewImg = (GetAssociateImageFunc)GetProcAddress(m_pKfbLib, "GetPriviewInfoPathFunc");
        CHECK_NULL( m_GetPreviewImg );

        m_GetLabelImage = (GetLableInfoPathFunc)dlsym(m_pKfbLib, "GetLableInfoPathFunc");
		// m_GetLabelImage = (GetAssociateImageFunc)GetProcAddress(m_pKfbLib, "GetLableInfoPathFunc");
        CHECK_NULL( m_GetLabelImage );

        m_GetImageDataRoiFunc = (GetImageDataRoiFunc)dlsym(m_pKfbLib, "GetImageDataRoiFunc");
        CHECK_NULL( m_GetImageDataRoiFunc );

        m_GetImageRGBDataStreamFunc = (GetImageRGBDataStreamFunc)dlsym(m_pKfbLib, "GetImageRGBDataStreamFunc");
        CHECK_NULL(m_GetImageRGBDataStreamFunc);
        printf("Succeed to Initialize KfbLib\n");
        return true;
    }

    bool InitImageFile( ImageInfoStruct* imgInfo, const std::string& sFilePath )
    {
        m_initImage(imgInfo, sFilePath.c_str() );
        return true;
    }

    char* GetImageStream( ImageInfoStruct* imgInfo, float fScale, int nImagePosX, int nImagePosY, int* nDataLength, unsigned char** pImage)
    {
        return m_GetImageStream(imgInfo, fScale, nImagePosX, nImagePosY, nDataLength, pImage);
    }

    bool UnInitImageFile( ImageInfoStruct* imgInfo )
    {
        m_UnInitImage(imgInfo );
        return 1;
    }

    bool GetHeaderInfo( ImageInfoStruct* imgInfo, int& iHeight, int& iWidth, int& iScale, float& fSpendTime, double& dScanTime, float& fImgCapRes,
                        int& iBlockSize)
    {
        m_GetHeaderInfo(imgInfo, &iHeight, &iWidth, &iScale, &fSpendTime, &dScanTime, &fImgCapRes, &iBlockSize);
        bool rtn = iWidth > 0 && iHeight > 0 && iScale > 0 && iBlockSize > 0;
        return rtn;
    }

    bool GetThumbImage( const std::string& sFileName, unsigned char** pImageData, int& iDataLength, int& iWidth, int& iHeight)
    {
        return m_GetThumbImg(sFileName.c_str(), pImageData, &iDataLength, &iWidth, &iHeight);
		// return GetAssociateImage_i(m_GetThumbImg, sFileName.c_str(), pImageData, iDataLength, iWidth, iHeight);
    }

    bool GetPreviewImage( const std::string& sFileName, unsigned char** pImageData, int& iDataLength, int& iWidth, int& iHeight)
    {
        return m_GetPreviewImg(sFileName.c_str(), pImageData, &iDataLength, &iWidth, &iHeight);
		// return GetAssociateImage_i(m_GetPreviewImg, sFileName.c_str(), pImageData, iDataLength, iWidth, iHeight);
    }

    bool GetLabelImage( const std::string& sFileName, unsigned char** pImageData, int& iDataLength, int& iWidth, int& iHeight)
    {
        return m_GetLabelImage(sFileName.c_str(), pImageData, &iDataLength, &iWidth, &iHeight);
		// return GetAssociateImage_i(m_GetLabelImage, sFileName.c_str(), pImageData, iDataLength, iWidth, iHeight);
    }

    bool GetImageRoiStream( ImageInfoStruct* imgInfo, float fScale, int nImagePosX, int nImagePosY,
        int nWidth, int nHeight, int *DataLength, unsigned char** pImage)
    {
        return m_GetImageDataRoiFunc(imgInfo, fScale, nImagePosX, nImagePosY, nWidth, nHeight, pImage, DataLength, true);
    }

    bool GetImageRoiRGBRawStream(ImageInfoStruct* imgInfo, float fScale, int nImagePosX, int nImagePosY,
        int nWidth, int nHeight, int *DataLength, unsigned char** pImage)
    {
        return m_GetImageRGBDataStreamFunc(imgInfo, fScale, nImagePosX, nImagePosY, DataLength, &nWidth, &nHeight, pImage);
    }

protected:
    template< typename tType >
    tType get_sym(const std::string& sSymName )
    {
        tType pFunc = (tType)dlsym( m_pKfbLib, sSymName.c_str() );
        char* pErr = dlerror();
        if ( pErr )
        {
            printf("Fail to get_sym : %s\n", pErr);
            return NULL;
        }
        return pFunc;
    }

    bool GetAssociateImage_i( GetAssociateImageFunc infoFunc, const std::string& sFileName,
                              unsigned char** pImageData, int& iDataLength, int& iWidth, int& iHeight)
    {
        if ( !infoFunc( sFileName.c_str(), pImageData, &iDataLength, &iWidth, &iHeight))
        {
            printf("Fail to get file : %s\n", sFileName.c_str());
            return false;
        }

        return true;
    }

private:
    KfbLib( const std::string& sLibFile )
        : m_sLibFile( sLibFile ), m_pKfbLib( NULL )
    {
    }


private:
    static KfbLib*   m_kfbLib;

private:
    std::string  m_sLibFile;
    void*        m_pKfbLib;         // Linux Version
    void*        m_pCurlLib;
    void*        m_pJpegLib;
	// HINSTANCE	 m_pKfbLib;     // Windows Version
};


KfbLib* KfbLib::m_kfbLib = NULL;

namespace {
	bool level_in_range(kfbslide_t* osr, int32_t level)
	{
		return (level >= 0 && level < osr->m_iLevelCount);
	}

	std::string to_string(float fin)
	{
		char cTmp[50] = { 0 };
		sprintf(cTmp, "%f", fin);
		return std::string(cTmp);
	}

	bool make_properties(kfbslide_t* osr)
	{
		typedef std::map<std::string, std::string >::iterator Iterator;
		std::map< std::string, std::string > mapPropertiesTmp;
		mapPropertiesTmp[OPENSLIDE_PROPERTY_NAME_MPP_X] = to_string(osr->m_fImgCapRes);
		mapPropertiesTmp[OPENSLIDE_PROPERTY_NAME_MPP_Y] = to_string(osr->m_fImgCapRes);
		mapPropertiesTmp[OPENSLIDE_PROPERTY_NAME_VENDOR] = "Kfbio";

		int iPropertiesNamesCnt = static_cast<int>(mapPropertiesTmp.size());
		osr->m_pPropertiesNames = new const char*[iPropertiesNamesCnt + 1];
		osr->m_pPropertiesNames[iPropertiesNamesCnt] = NULL;
		int index = 0;
		for (Iterator itor = mapPropertiesTmp.begin(); itor != mapPropertiesTmp.end(); ++itor)
		{
			osr->m_pPropertiesNames[index++] = itor->first.c_str();
		}
		osr->m_mapProperties.swap(mapPropertiesTmp);
		return true;
	}

	bool make_associate_images(kfbslide_t* osr)
	{
		KfbLib* pKfbLib = KfbLib::Instance();
		AssociateImage thumb_img;
        if (pKfbLib->GetThumbImage(osr->m_sFileName, &thumb_img.m_ImagePixel, thumb_img.m_iDataLength,
            thumb_img.m_iWidth, thumb_img.m_iHeight) && thumb_img.m_iDataLength > 0)
		{
			//printf("thumbnail DataLength : %d\n", thumb_img.m_iDataLength);
			osr->m_mapAssociateImgs["thumbnail"] = thumb_img;
		}

		AssociateImage preview_img;
        if (pKfbLib->GetPreviewImage(osr->m_sFileName, &preview_img.m_ImagePixel, preview_img.m_iDataLength,
            preview_img.m_iWidth, preview_img.m_iHeight) && preview_img.m_iDataLength > 0)
		{
			//printf("preview DataLength : %d\n", preview_img.m_iDataLength);
			osr->m_mapAssociateImgs["macro"] = preview_img;
		}

		AssociateImage label_img;
        if (pKfbLib->GetLabelImage(osr->m_sFileName, &label_img.m_ImagePixel, label_img.m_iDataLength,
            label_img.m_iWidth, label_img.m_iHeight) && label_img.m_iDataLength > 0)
		{
			//printf("label DataLength : %d\n", label_img.m_iDataLength);
			osr->m_mapAssociateImgs["label"] = label_img;
		}

		int iAssociateImgsCnt = static_cast<int>(osr->m_mapAssociateImgs.size());
		if (0 < iAssociateImgsCnt)
		{
			osr->m_pAssociateImageNames = new const char*[iAssociateImgsCnt + 1];
			osr->m_pAssociateImageNames[iAssociateImgsCnt] = NULL;
			int index = 0;
			for (std::map< std::string, AssociateImage >::iterator itor = osr->m_mapAssociateImgs.begin();
				itor != osr->m_mapAssociateImgs.end(); ++itor)
			{
				osr->m_pAssociateImageNames[index++] = itor->first.c_str();
			}
		}
		return true;
	}
}

const char* kfbslide_detect_vendor(const char* filename)
{
    return "kfbio";
}

kfbslide_t* kfbslide_open(const char* filename)
{
    KfbLib* pkfb_lib = KfbLib::Instance();
    if ( !pkfb_lib )
    {
        printf("pkfb_lib is NULL");
        return NULL;
    }
    kfbslide_t* pkfbSlide = new kfbslide_t();
    if ( !pkfbSlide )
    {
        printf("[kfbslide.so] Fail to allocate kfbslide_t\n");
        return NULL;
    }
    printf("[kfbslide.so] File name is : %s\n", filename);
    if ( !pkfb_lib->InitImageFile( &pkfbSlide->m_imgInfo, filename) )
    {
        printf("[kfbslide.so] Fail to InitImageFile\n");
        delete pkfbSlide;
        return NULL;
    }
    printf("[kfbslide.so] Data File Ptr Value [After]: %d\n", pkfbSlide->m_imgInfo.DataFilePTR);

    int iScale = 0, iWidth = 0, iHeight = 0;
    if ( !pkfb_lib->GetHeaderInfo( &pkfbSlide->m_imgInfo, iHeight, iWidth, iScale,
                                   pkfbSlide->m_fSpendTime, pkfbSlide->m_dScanTime, pkfbSlide->m_fImgCapRes, pkfbSlide->m_iBlockSize) )
    {
        printf("[kfbslide.so] Fail to GetHeaderInfo\n");
		printf("Parameters: %d, %d, %d, %d\n", iHeight, iWidth, iScale, pkfbSlide->m_iBlockSize);
        delete pkfbSlide;
        return NULL;
    }

    printf("[kfbslide.so] Succeed to Get HeaderInfo\n" );
    pkfbSlide->m_sFileName = std::string(filename);
    pkfbSlide->m_fScale = iScale;
    pkfbSlide->m_iWidth = iWidth;
    pkfbSlide->m_iHeight = iHeight;
    //float flog = log(pkfbSlide->m_fScale) / log(2);
    //int iLevelCnt = int(flog);
    int iMaxSize = iWidth > iHeight ? iWidth : iHeight;
    int iLevelCnt = log( iMaxSize ) / log(2);
    iLevelCnt = iLevelCnt > MaxKFBLevel ? MaxKFBLevel : iLevelCnt;

    pkfbSlide->m_iLevelCount = iLevelCnt + 1;
    pkfbSlide->m_pDimensions = new dimension_t[pkfbSlide->m_iLevelCount];
    pkfbSlide->m_pDimensions[0].w = pkfbSlide->m_iWidth;
    pkfbSlide->m_pDimensions[0].h = pkfbSlide->m_iHeight;
    pkfbSlide->m_pDimensions[0].downsample = 1.0;
    for (int iLevel = 1; iLevel <= iLevelCnt; ++iLevel)
    {
        pkfbSlide->m_pDimensions[iLevel].w = pkfbSlide->m_pDimensions[iLevel - 1].w / 2;
        pkfbSlide->m_pDimensions[iLevel].h = pkfbSlide->m_pDimensions[iLevel - 1].h / 2;
        pkfbSlide->m_pDimensions[iLevel].downsample = pkfbSlide->m_pDimensions[ iLevel - 1 ].downsample * 2;
    }
    /*
    if( flog - iLevelCnt >= 0.5 )
    {
        pkfbSlide->m_pDimensions[ iLevelCnt + 1 ].w = pkfbSlide->m_pDimensions[iLevelCnt].w / 2;
        pkfbSlide->m_pDimensions[ iLevelCnt + 1 ].h = pkfbSlide->m_pDimensions[iLevelCnt].h / 2;
        pkfbSlide->m_pDimensions[ iLevelCnt + 1 ].downsample = pkfbSlide->m_pDimensions[iLevelCnt].downsample * 2.0;
    }
    */

    //printf("Line %d\n", __LINE__);
    // make_properties(pkfbSlide);
    // make_associate_images( pkfbSlide );
    return pkfbSlide;
}


void kfbslide_set_attrs( kfbslide_t* pkfbSlide, int iHeight, int iWidth,
                         int iScale, float fSpendTime, double dScanTime, float fImgCapRes, int iBlockSize)
{
    pkfbSlide->m_iHeight = iHeight;
    pkfbSlide->m_iWidth  = iWidth;
    pkfbSlide->m_fScale = iScale;
    pkfbSlide->m_fSpendTime = fSpendTime;
    pkfbSlide->m_dScanTime  = dScanTime;
    pkfbSlide->m_fImgCapRes = fImgCapRes;
    pkfbSlide->m_iBlockSize = iBlockSize;
    // float flog = log(pkfbSlide->m_fScale) / log(2);
    // int iLevelCnt = int(flog);
    int iMaxSize = iWidth > iHeight ? iWidth : iHeight;
    int iLevelCnt = log( iMaxSize ) / log(2);
    iLevelCnt = iLevelCnt > MaxKFBLevel ? MaxKFBLevel : iLevelCnt;

    pkfbSlide->m_iLevelCount = iLevelCnt + 1;
    pkfbSlide->m_pDimensions = new dimension_t[pkfbSlide->m_iLevelCount];
    pkfbSlide->m_pDimensions[0].w = pkfbSlide->m_iWidth;
    pkfbSlide->m_pDimensions[0].h = pkfbSlide->m_iHeight;
    pkfbSlide->m_pDimensions[0].downsample = 1.0;
    for (int iLevel = 1; iLevel <= iLevelCnt; ++iLevel)
    {
        pkfbSlide->m_pDimensions[iLevel].w = ceil(float(pkfbSlide->m_pDimensions[iLevel - 1].w) / 2);
        pkfbSlide->m_pDimensions[iLevel].h = ceil(float(pkfbSlide->m_pDimensions[iLevel - 1].h) / 2);
        pkfbSlide->m_pDimensions[iLevel].downsample = pkfbSlide->m_pDimensions[ iLevel - 1 ].downsample * 2;
    }

    // printf("Line %d\n", __LINE__);
    make_properties(pkfbSlide);
    make_associate_images( pkfbSlide );
    //return pkfbSlide;
}

int32_t kfbslide_get_level_count(kfbslide_t* osr)
{
    return osr->m_iLevelCount;
}

void kfbslide_get_level0_dimensions(kfbslide_t* osr, int64_t* w, int64_t* h)
{
    *w = osr->m_iWidth;
    *h = osr->m_iHeight;
}

void kfbslide_get_level_dimensions(kfbslide_t* osr, int32_t level, int64_t *w, int64_t* h)
{
    if ( !level_in_range(osr, level))
    {
        return;
    }
    *w = osr->m_pDimensions[level].w;
    *h = osr->m_pDimensions[level].h;
}

double kfbslide_get_level_downsample( kfbslide_t* osr, int32_t level)
{
    if (!level_in_range(osr, level))
    {
        return 0.0;
    }
    return osr->m_pDimensions[level].downsample;
}

int32_t kfbslide_get_best_level_for_downsample( kfbslide_t* osr, double downsample)
{
    if ( downsample < osr->m_pDimensions[0].downsample)
    {
        return 0;
    }

    for (int i = 0; i < osr->m_iLevelCount; ++i)
    {
        if ( downsample < osr->m_pDimensions[i].downsample)
        {
            return i - 1;
        }
    }
    return osr->m_iLevelCount - 1;
}

static int kfb_img_index = 0;

bool kfbslide_read_region( kfbslide_t* osr, int32_t level, int64_t x, int64_t y, int* iDataLength, unsigned char** dest)
{
    //printf("Enter kfbslide_read_region\n");
    if (!level_in_range(osr, level))
    {
        printf("Level %d is out of range\n", level);
        return false;
    }
    //printf("%d\n", __LINE__);
    KfbLib* pkfbLib = KfbLib::Instance();
    float fScale_output =  float( osr->m_fScale / osr->m_pDimensions[level].downsample);
    x = int( x / (osr->m_pDimensions[level].downsample * 256 ) ) * 256;
    y = int( y / (osr->m_pDimensions[level].downsample * 256 ) ) * 256;

    //printf("%d\n", __LINE__);
    unsigned char* pOutputPixel = NULL;
    pkfbLib->GetImageStream( &osr->m_imgInfo, fScale_output, x, y, iDataLength, &pOutputPixel);
    // printf("img index : %d, level : %d, Scale : %f, x : %d, y: %d, dataLen : %d\n", kfb_img_index++, int(level), fScale_output, int(x), int(y), *iDataLength);
    if ( iDataLength <= 0 )
    {
        printf("Fail to Get Image Stream");
        return false;
    }
    *dest = pOutputPixel;
    return true;
}

bool kfbslide_get_image_roi_stream(kfbslide_t* osr, int32_t level, 
    int64_t x, int64_t y, int64_t width, int64_t height, int* data_length, unsigned char** img_jpeg_stream)
{
    if (!level_in_range(osr, level))
    {
        printf("Level %d is out of range\n", level);
        return false;
    }
    KfbLib* pkfbLib = KfbLib::Instance();
    float fScale_output =  float( osr->m_fScale / osr->m_pDimensions[level].downsample);
    pkfbLib->GetImageRoiStream(&osr->m_imgInfo, fScale_output, x, y, width, height, data_length, img_jpeg_stream);
    if(data_length <= 0)
    {
        printf("Fail to Get Image Roi Stream");
        return false;
    }
    return true;
}

bool kfbslide_get_image_roi_rgb_rawdata(kfbslide_t* osr, int32_t level, 
    int64_t x, int64_t y, int64_t width, int64_t height, int* data_length, unsigned char** img_rgb_stream)
{
    if (!level_in_range(osr, level))
    {
        printf("Level %d is out of range\n", level);
        return false;
    }
    KfbLib* pkfbLib = KfbLib::Instance();
    float fScale_output =  float( osr->m_fScale / osr->m_pDimensions[level].downsample);
    pkfbLib->GetImageRoiRGBRawStream(&osr->m_imgInfo, fScale_output, x, y, width, height, data_length, img_rgb_stream);
    if(data_length <= 0)
    {
        printf("Fail to Get Image Roi Stream");
        return false;
    }
    return true;
}

void kfbslide_close(kfbslide_t* osr)
{
    KfbLib* pkfbLib = KfbLib::Instance();
    if ( !pkfbLib->UnInitImageFile(&osr->m_imgInfo) )
    {
        printf("Fail to UnInitImageFile");
        return;
    }
    delete osr;
}

void kfbslide_get_error(kfbslide_t *osr)
{
	return;
}

const char* const* kfbslide_get_property_names( kfbslide_t* osr)
{
    return osr->m_pPropertiesNames;
}

const char* kfbslide_get_property_value( kfbslide_t* osr, const char* name)
{
    std::map< std::string, std::string>::iterator itor = osr->m_mapProperties.find( name );
    if ( itor == osr->m_mapProperties.end())
    {
        return NULL;
    }
    return itor->second.c_str();
}

const char* const* kfbslide_get_associated_image_names( kfbslide_t* osr)
{
    return osr->m_pAssociateImageNames;
}

void kfbslide_get_associated_image_dimensions( kfbslide_t* osr,
        const char* name, int64_t* w, int64_t* h, int* iDatLength)
{
    std::map< std::string, AssociateImage >::iterator itor = osr->m_mapAssociateImgs.find( name );
    if ( itor == osr->m_mapAssociateImgs.end() )
    {
        printf("%s is not in AssociateImages", name);
        return;
    }

    *w = itor->second.m_iWidth;
    *h = itor->second.m_iHeight;
    *iDatLength = itor->second.m_iDataLength;
    return;
}

void kfbslide_read_associated_image( kfbslide_t* osr,
                                     const char* name, unsigned char** dest)
{
    std::map< std::string, AssociateImage >::iterator itor = osr->m_mapAssociateImgs.find(name);
    if ( itor == osr->m_mapAssociateImgs.end() )
    {
        printf("Fail to read associate image %s\n", name);
        return;
    }
    *dest = itor->second.m_ImagePixel;
}


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

void
WsiPredictor::predict(kfbslide_t * p_wsi)
{
    //--------
	// Check the bad input pointer and get the size of the WSI
	//--------  
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
    
    kfbslide_get_level0_dimensions(p_wsi, &m_sz_wsi_width, &m_sz_wsi_height);
    // CHECK(NULL == openslide_get_error(p_wsi))
    //     << "Get level 0 dimension error using openslide library.";    
    
    kfb_calculate_tile_nums(p_wsi);
    
    //--------
	// Calculate the size of the prob. map and allocate memory for it
	//--------  
    int width  = m_num_col * m_sz_tile_output;
    int height = m_num_row * m_sz_tile_output;
    m_probmap  = cv::Mat(height, width, CV_32FC1, cv::Scalar(0));
    std::cout<<"Start to process."<<std::endl;
    
    //--------
	// Start the single producer and multi consumer queue
	//--------  
    moodycamel::BlockingConcurrentQueue<QueueItem *> queue;
    std::thread tile_processors[m_gpu_ids.size()];
    std::thread tile_reader(&WsiPredictor::kfb_tile_reader, 
                            this,
                            p_wsi,
                            &queue);
    
    for (size_t i = 0; i < m_gpu_ids.size(); ++i)
    {
        tile_processors[i] = std::thread(&WsiPredictor::kfb_tile_processor, 
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

void
WsiPredictor::save_heatmap(
    std::string file_path,
    double      sample_rate
)
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
    cv::Mat save_map;
    output_map.convertTo(save_map, CV_8UC3,255);
    cv::imwrite(file_path, save_map);
    
    //const unsigned int shape[] = {height, width};
    //cnpy::npy_save(file_path, (float*)output_map.data, shape, 2, "w");
    // cnpy::npy_save(file_path, (float*)output_map.data, {height, width}, "w");
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
    // CHECK(NULL == openslide_get_error(p_wsi))
    //     << "Read region error with openslide library.";
    
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

void
WsiPredictor::kfb_read_region_from_wsi(
    kfbslide_t *        p_wsi,
    cv::Mat &           result,                     
    const int64_t       x,
    const int64_t       y,
    const int32_t       level,
    const int64_t       w,
    const int64_t       h)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
    
    // uint32_t * p_src = (uint32_t *)malloc(w * h * 4);
    // CHECK(p_src) << "Unable to allocate enough memory";
    // openslide_read_region(p_wsi, p_src, x, y, level, w, h);
    
    int iDataLength = 0;
    uchar* p_src = nullptr;
    kfbslide_get_image_roi_stream(p_wsi, level, x, y, w, h, &iDataLength, &p_src);
    // CHECK(NULL == openslide_get_error(p_wsi))
    //     << "Read region error with openslide library.";
    cv::Mat rawdata( 1, iDataLength, CV_8UC1, (void*)p_src );
    cv::Mat jpgmat = cv::imdecode(rawdata, CV_LOAD_IMAGE_COLOR);
    cv::cvtColor(jpgmat, result, CV_RGB2BGR);
    delete[] p_src;
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

void 
WsiPredictor::kfb_preprocessing(
    kfbslide_t *        p_wsi,
    cv::Mat &           result,
    const int64_t       width,
    const int64_t       height)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
        
    cv::Mat img_ov(height, width, CV_8UC3);
    kfb_read_region_from_wsi( p_wsi, 
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

void 
WsiPredictor::kfb_calculate_tile_nums(
    kfbslide_t *    p_wsi)
{
    CHECK(NULL != p_wsi)
        << "Please pass in a valid OpenSlide object pointer.";
        
    //--------
	// Calculate the size of the overview level
	//-------- 
    int level_count = kfbslide_get_level_count(p_wsi);
    int64_t width, height;
    m_lvl_ov = level_count - 1;
    kfbslide_get_level_dimensions(p_wsi, m_lvl_ov, &width, &height);
    
    //cout << level_count << '\t' << m_lvl_ov << endl;
    m_factor_w = (float)m_sz_wsi_width / (float) width;
    m_factor_h = (float)m_sz_wsi_height / (float) height;
    
    //--------
	// Threashold the overview level 
	//-------- 
    cv::Mat img_thresh(height, width, CV_8UC1);
    kfb_preprocessing(p_wsi, img_thresh, width, height);
    
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

void
WsiPredictor::kfb_tile_reader(
    kfbslide_t *                                        p_wsi,
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
                    kfb_read_region_from_wsi(p_wsi,
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
                    kfb_read_region_from_wsi(p_wsi,
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

void
WsiPredictor::kfb_tile_processor(
    kfbslide_t *                                        p_wsi,
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
