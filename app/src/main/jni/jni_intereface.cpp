// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/native_window_jni.h>
#include <android/native_window.h>

#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

//#include <platform.h>
//#include <benchmark.h>
#include "sam.h"

#include "ndkcamera.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include<chrono>

#if __ARM_NEON
#include <arm_neon.h>
#endif // __ARM_NEON


static SAM* g_SAM = 0;
static std::string img_truck= ""; 





static int draw_fps(cv::Mat& rgb)
{
    // resolve moving average
   float avg_fps = 0.f;
   {
       static double t0 = 0.f;
       static float fps_history[10] = {0.f};

    //    double t1 = ncnn::get_current_time();
       auto now = std::chrono::system_clock::now();
        // 将时间戳转换为毫秒数
       auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
       double t1 = now_ms.time_since_epoch().count();

       if (t0 == 0.f)
       {
           t0 = t1;
           return 0;
       }

       float fps = 1000.f / (t1 - t0);
       t0 = t1;

       for (int i = 9; i >= 1; i--)
       {
           fps_history[i] = fps_history[i - 1];
       }
       fps_history[0] = fps;

       if (fps_history[9] == 0.f)
       {
           return 0;
       }

       for (int i = 0; i < 10; i++)
       {
           avg_fps += fps_history[i];
       }
       avg_fps /= 10.f;
   }

    char text[32];
   sprintf(text, "FPS=%.2f", avg_fps);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int y = 0;
    int x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                    cv::Scalar(255, 255, 255), -1);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

    return 0;
}


class MyNdkCamera : public NdkCameraWindow
{
public:
    virtual void on_image_render(cv::Mat& rgb) const;
};

void MyNdkCamera::on_image_render(cv::Mat& rgb) const
{
    // nanodet
    {
        //  ncnn::MutexLockGuard g(lock);

        if (g_SAM)
        {
            g_SAM->AutoPredict(rgb);

        }
//         else
//         {
//             draw_unsupported(rgb);
//         }
    }

    draw_fps(rgb);
}

static MyNdkCamera* g_camera = 0;

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnLoad");

    g_camera = new MyNdkCamera;

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "JNI_OnUnload");

    {
        // ncnn::MutexLockGuard g(lock);

        delete g_SAM;
        g_SAM = 0;
    }

    delete g_camera;
    g_camera = 0;
}

// public native boolean loadModel(AssetManager mgr, int modelid, int cpugpu);
JNIEXPORT jboolean JNICALL Java_com_tencent_mSAM_mSAM_loadModel(JNIEnv* env, jobject thiz, jstring name, jint modelid, jint cpugpu)
{
    if (modelid < 0 || modelid > 6 || cpugpu < 0 || cpugpu > 1)
    {
        return JNI_FALSE;
    }

    // AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
//    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "loadModel %p", mgr);

    bool use_gpu = (int)cpugpu == 1;
    int width_point[5]= {1,2,3,4,5};

    // reload
    {
        //  ncnn::MutexLockGuard g(lock);

//         if (use_gpu)
//         {
//             // no gpu
//             delete g_SAM;
//             g_SAM = 0;
//         }
//         else
        {
            if (!g_SAM){
                char model_name[256];
                const char *pathTemp = env->GetStringUTFChars(name, 0);
                std::string modelPath_base = pathTemp;
                std::string modeltype_decoder = pathTemp;
                std::string modeltype_encoder = pathTemp;
                modeltype_encoder= modelPath_base+ "mobile_embed.mnn";
                modeltype_decoder= modelPath_base+ "mobile_segment.mnn";
                //  img_truck=  modelPath_base+"truck.jpg";

                g_SAM = new SAM();
                g_SAM->Init(modeltype_decoder, modeltype_encoder, use_gpu, width_point[modelid]);
            }
        }
    }

    return JNI_TRUE;
}

// public native boolean openCamera(int facing);
JNIEXPORT jboolean JNICALL Java_com_tencent_mSAM_mSAM_openCamera(JNIEnv* env, jobject thiz, jint facing)
{
    if (facing < 0 || facing > 1)
        return JNI_FALSE;

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "openCamera %d", facing);

    g_camera->open((int)facing);

    return JNI_TRUE;
}

// public native boolean closeCamera();
JNIEXPORT jboolean JNICALL Java_com_tencent_mSAM_mSAM_closeCamera(JNIEnv* env, jobject thiz)
{
    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "closeCamera");

    g_camera->close();

    return JNI_TRUE;
}

// public native boolean setOutputWindow(Surface surface);
JNIEXPORT jboolean JNICALL Java_com_tencent_mSAM_mSAM_setOutputWindow(JNIEnv* env, jobject thiz, jobject surface)
{
    ANativeWindow* win = ANativeWindow_fromSurface(env, surface);

    __android_log_print(ANDROID_LOG_DEBUG, "ncnn", "setOutputWindow %p", win);

    g_camera->set_window(win);

    return JNI_TRUE;
}

}
