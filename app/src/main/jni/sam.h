#ifndef SAM_H
#define SAM_H

#include <opencv2/core/core.hpp>

#include <MNN/ImageProcess.hpp>
#include <MNN/expr/Module.hpp>
#include <MNN/expr/Executor.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <cv/cv.hpp>

#include <android/log.h>

using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;



class SAM
{
public:
    SAM();

    int Init(std::string& modeltype_decoder, std::string& modeltype_encoder, bool use_gpu = false, int p_point_width= 3);

    int AutoPredict(const cv::Mat& bgr);
private:

    int prompt_point_width= 3;
    std::shared_ptr<Module> embed;
    std::shared_ptr<Module> sam;

    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    int image_w;
    int image_h;
    int in_w;
    int in_h;

};

#endif
