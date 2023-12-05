#include "sam.h"

//#include <opencv2/core/core.hpp>
//#include <opencv2/imgproc/imgproc.hpp>
using namespace MNN;
using namespace MNN::Express;
using namespace MNN::CV;
//#include "cpu.h"

void get_grid_points(std::vector<float>& points_xy_vec, int n_per_side);
void get_grid_points(std::vector<float>& points_xy_vec, int n_per_side)
{
    float offset = 1.f / (2 * n_per_side);

    float start = offset;
    float end = 1 - offset;
    float step = (end - start) / (n_per_side - 1);

    std::vector<float> points_one_side;
    for (int i = 0; i < n_per_side; ++i) {
        points_one_side.push_back(start + i * step);
    }

    points_xy_vec.resize(n_per_side * n_per_side * 2);
    for (int i = 0; i < n_per_side; ++i) {
        for (int j = 0; j < n_per_side; ++j) {
            points_xy_vec[i * n_per_side * 2 + 2 * j + 0] = points_one_side[j];
            points_xy_vec[i * n_per_side * 2 + 2 * j + 1] = points_one_side[i];
        }
    }
}
 
SAM::SAM()
{
    // blob_pool_allocator.set_size_compare_ratio(0.f);
    // workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int SAM::Init(std::string& modeltype_decoder, std::string& modeltype_encoder, bool use_gpu, int p_point_width)
{

    // char modelpath_decode[256];
    // char modelpath_encode[256];
    // sprintf(modelpath_encode, "%s.mnn", modeltype_encoder);
    // sprintf(modelpath_decode, "%s.mnn", modeltype_decoder);

//    prompt_point_width= p_point_width;

    int thread = 4;
    int precision = 1;
    int forwardType = MNN_FORWARD_CPU;
    if(use_gpu){
        forwardType= MNN_FORWARD_OPENCL;
    }

    MNN::ScheduleConfig sConfig;
    sConfig.type = static_cast<MNNForwardType>(forwardType);
    sConfig.numThread = thread;
    MNN::BackendConfig bConfig;
    bConfig.precision = static_cast<MNN::BackendConfig::PrecisionMode>(precision);
    sConfig.backendConfig = &bConfig;
    std::shared_ptr<Executor::RuntimeManager> rtmgr = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
    if(rtmgr == nullptr) {
        MNN_ERROR("Empty RuntimeManger\n");
        return 0;
    }
    // rtmgr->setCache(".cachefile");
    __android_log_print(ANDROID_LOG_DEBUG, "mnn path", " %s %s", modeltype_encoder.c_str(), modeltype_decoder.c_str());

    // init 赋值传到class内部，不然下面调用 必然报错
    embed= std::shared_ptr<Module>(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, modeltype_encoder.c_str(), rtmgr));
    sam= std::shared_ptr<Module>(Module::load(
        {"point_coords", "point_labels", "image_embeddings", "has_mask_input", "mask_input", "orig_im_size"},
        {"iou_predictions", "low_res_masks", "masks"}, modeltype_decoder.c_str(), rtmgr));

    return 1;
}


int SAM::AutoPredict(const cv::Mat& bgr)
{

    auto dims = bgr.channels();
    int origin_h = bgr.rows;
    int origin_w = bgr.cols;
    int length = 1024;
    int new_h, new_w;
    if (origin_h > origin_w) {
        new_w = round(origin_w * (float)length / origin_h);
        new_h = length;
    } else {
        new_h = round(origin_h * (float)length / origin_w);
        new_w = length;
    }
    float scale_w = (float)new_w / origin_w;
    float scale_h = (float)new_h / origin_h;
    // mat 转 mnn
    // 获取图像的宽、高和通道数

    int width = bgr.cols;
    int height = bgr.rows;
    int channels = bgr.channels();
//    __android_log_print(ANDROID_LOG_DEBUG, "mnn path", "w %d h %d", width, height);

    // 将图像转换为 unsigned char* 格式
    unsigned char* imgData = bgr.data;
    // 创建一个 Tensor 用于存储图像数据
    MNN::Tensor::DimensionType dim_type = MNN::Tensor::TENSORFLOW;
    std::vector<int> dims1{1, height, width, channels};
    auto tensor = MNN::Tensor::create<u_int8_t>(dims1, NULL, dim_type);
    auto tensorData = tensor->host<u_int8_t>();
    auto tensor_size = tensor->size();
    std::memcpy(tensorData, imgData, tensor_size);
    // 创建一个 MNN::Express::VARP，将张量封装为表达式的形式
    auto original_image = MNN::Express::_Const(tensorData,dims1,MNN::Express::NHWC,halide_type_of<u_int8_t>());
    auto image = _Squeeze(original_image,{0});

    auto input_var = resize(image, Size(new_w, new_h), 0, 0, INTER_LINEAR, -1, {123.675, 116.28, 103.53}, {1/58.395, 1/57.12, 1/57.375});
    std::vector<int> padvals { 0, length - new_h, 0, length - new_w, 0, 0 };
    auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
    input_var = _Pad(input_var, pads, CONSTANT);
    input_var = _Unsqueeze(input_var, {0});

//    __android_log_print(ANDROID_LOG_DEBUG, "mnn path before embed", " dim %d %d %d %d", input_var->getInfo()->dim[0], input_var->getInfo()->dim[1], input_var->getInfo()->dim[2], input_var->getInfo()->dim[3]);
    input_var= _Convert(input_var, MNN::Express::NC4HW4);
//    __android_log_print(ANDROID_LOG_DEBUG, "mnn path before embed", " dim %d %d %d %d", input_var->getInfo()->dim[0], input_var->getInfo()->dim[1], input_var->getInfo()->dim[2], input_var->getInfo()->dim[3]);

//    __android_log_print(ANDROID_LOG_DEBUG, "mnn path names", " dim %s %s %s", embed->getInfo()->version.c_str(), embed->getInfo()->inputNames[0].c_str(), embed->getInfo()->outputNames[0].c_str());


    auto outputs = embed->onForward({input_var});
    auto image_embedding = _Convert(outputs[0], NCHW);

//    __android_log_print(ANDROID_LOG_DEBUG, "mnn path after_embed", " w %d h %d", width, height);

    auto build_input = [](std::vector<float> data, std::vector<int> shape) {
        return _Const(static_cast<void*>(data.data()), shape, NCHW, halide_type_of<float>());
    };
    // build inputs
    std::vector<float> points;
    std::vector<float> labels;
    int n_per_side= prompt_point_width;
    float region_ratio = 0.5;
    get_grid_points(points, n_per_side);
    for(int i = 0; i < n_per_side; ++i) {
        for(int j = 0; j < n_per_side; ++j) {
            int x= i * n_per_side * 2 + 2 * j;
            points[x]= points[x]*origin_w* region_ratio + origin_w*region_ratio*0.5;
            points[x+1]= points[x+1]*origin_h* region_ratio + origin_h*region_ratio*0.5;
        }
    }
    // std::vector<float> points = {300, 300};
    std::vector<float> scale_points;
    for(int i = 0; i < n_per_side; ++i) {
        for(int j = 0; j < n_per_side; ++j) {
            int x= i * n_per_side * 2 + 2 * j;
            scale_points.push_back(points[x]* scale_w);
            scale_points.push_back(points[x+1]* scale_h);
            scale_points.push_back(0);
            scale_points.push_back(0);
            labels.push_back(1);
            labels.push_back(-1);
        }
    }
    // for (int i = 0; i < scale_points.size() / 2; i++) {
    //     scale_points[2 * i] = scale_points[2 * i] * scale_w;
    //     scale_points[2 * i + 1] = scale_points[2 * i + 1] * scale_h;
    // }
    // scale_points.push_back(0);
    // scale_points.push_back(0);
    auto point_coords = build_input(scale_points, {1, 2*n_per_side*n_per_side, 2});
    auto point_labels = build_input(labels, {1, 2*n_per_side*n_per_side});
    // auto point_coords = build_input(scale_points, {1, 2, 2});
    // auto point_labels = build_input({1, -1}, {1, 2});
    auto orig_im_size = build_input({static_cast<float>(origin_h), static_cast<float>(origin_w)}, {2});
    auto has_mask_input = build_input({0}, {1});
    std::vector<float> zeros(256*256, 0.f);
    auto mask_input = build_input(zeros, {1, 1, 256, 256});

//    __android_log_print(ANDROID_LOG_DEBUG, "mnn path image_embedding ", " dim %d %d %d %d", image_embedding->getInfo()->dim[0], image_embedding->getInfo()->dim[1], image_embedding->getInfo()->dim[2], image_embedding->getInfo()->dim[3]);
//    __android_log_print(ANDROID_LOG_DEBUG, "mnn path sam names", " dim %s %s %s", sam->getInfo()->version.c_str(), sam->getInfo()->inputNames[0].c_str(), sam->getInfo()->outputNames[0].c_str());


    auto output_vars = sam->onForward({point_coords, point_labels, image_embedding, has_mask_input, mask_input, orig_im_size});

    // __android_log_print(ANDROID_LOG_DEBUG, "mnn path after sam ", " dim %s %s %s", sam->getInfo()->version.c_str(), sam->getInfo()->inputNames[0].c_str(), sam->getInfo()->outputNames[0].c_str());

    auto masks = _Convert(output_vars[2], NCHW);
    auto scores = _Convert(output_vars[0], NCHW);

    scores= _Squeeze(scores, {0});
    std::vector<std::pair<float, int>> scores_vec;
    auto outputsize= scores->getInfo()->size;
    auto outputptr= scores->readMap<float>();
    for (int i = 0; i < outputsize; ++i) {
        scores_vec.push_back(std::pair<float, int>(outputptr[i], i));
    }
    std::sort(scores_vec.begin(), scores_vec.end(), std::greater<std::pair<float, int>>());
    float pred_iou_thresh=0.8f;
    float mask_threshold =0 ;
    if (scores_vec[0].first > pred_iou_thresh) {
        int ch = scores_vec[0].second;
        masks = _Gather(_Squeeze(masks, {0}), _Scalar<int>(ch));
        masks = _Greater(masks, _Scalar(mask_threshold));
        masks = _Reshape(masks, {origin_h, origin_w, 1});

        std::vector<int> color_vec {0, 153, 255};
        auto color = _Const(static_cast<void*>(color_vec.data()), {1, 1, 3}, NCHW, halide_type_of<int>());
        float w1= 0.6;
        float w2= 0.4;
        auto alpha= _Const(static_cast<void*>(&w1), {1, 1, 1}, NCHW, halide_type_of<float>());
        auto beta= _Const(static_cast<void*>(&w2), {1, 1, 1}, NCHW, halide_type_of<float>());

        image = _Cast<uint8_t>(_Cast<int>(_Cast<float>(image) * alpha) + _Cast<int>(_Cast<float>(masks) * _Cast<float>(color) * beta));
        for (int i = 0; i < points.size() / 2; i++) {
            float x = points[2 * i];
            float y = points[2 * i + 1];
            circle(image, {x, y}, 6, {0, 0, 255}, 2);
        }
    }


    // 4. postprocess: draw mask and point
    // MobileSam has multi channel masks, get first
    // float mask_threshold =0 ;
    // masks = _Gather(_Squeeze(masks, {0}), _Scalar<int>(0));
    // masks = _Greater(masks, _Scalar(mask_threshold));
    // masks = _Reshape(masks, {origin_h, origin_w, 1});
    // std::vector<int> color_vec {30, 144, 255};
    // auto color = _Const(static_cast<void*>(color_vec.data()), {1, 1, 3}, NCHW, halide_type_of<int>());

//     image = _Cast<uint8_t>(_Cast<int>(image) + masks * color);

// //    __android_log_print(ANDROID_LOG_DEBUG, "mnn path plot ", " image dim %d %d %d %d", image->getInfo()->dim[0], image->getInfo()->dim[1], image->getInfo()->dim[2], image->getInfo()->dim[3]);

//     for (int i = 0; i < points.size() / 2; i++) {
//         float x = points[2 * i];
//         float y = points[2 * i + 1];
//         circle(image, {x, y}, 10, {0, 0, 255}, 5);
//     }

    // __android_log_print(ANDROID_LOG_DEBUG, "mnn path after plot ", " dim %s %s %s", sam->getInfo()->version.c_str(), sam->getInfo()->inputNames[0].c_str(), sam->getInfo()->outputNames[0].c_str());

    const uint8_t* dataPtr= image->readMap<u_int8_t>();
    memcpy(bgr.data, dataPtr, width*height*channels* sizeof(uint8_t));
    
    return 0;
}
