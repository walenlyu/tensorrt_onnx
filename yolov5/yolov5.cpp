#include <fstream>
#include <map>
#include <chrono>
#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "cuda_runtime_api.h"
#include "logging.h"
#include <opencv2/opencv.hpp>
#include<dirent.h>
#include"calibrator.h"

#define CHECK(status) \
    do\
    {\
        auto ret = (status);\
        if (ret != 0)\
        {\
            std::cerr << "Cuda failure: " << ret << std::endl;\
            abort();\
        }\
    } while (0)

#define USE_INT8  # FP32 FP16 INT8
// stuff we know about the network and the input/output blobs
static const int INPUT_H = 640;
static const int INPUT_W = 640;
static const int BBOX_SIZE = 25200;
static const int NUM_CLASS = 80;
static const int OUTPUT_SIZE = 25200*85;
static const float CONF_THRESH = 0.25;
static const float CLASS_THRESH = 0.25;
static const float IOU_THRESH = 0.4;

const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "outputs";

const float ANCHORS[3][6] = {{10.0, 13.0, 16.0, 30.0, 33.0, 23.0}, {30.0, 61.0, 62.0, 45.0, 59.0, 119.0},{116.0, 90.0, 156.0, 198.0, 373.0, 326.0}};
const float STRIDES[3] = { 8.0, 16.0, 32.0 };


using namespace nvinfer1;

static Logger gLogger;

struct  Detection
{
    int index;
    float score;
    std::vector<float> bbox;
};

bool cmp(Detection& detection1, Detection& detection2){
    return detection1.score > detection2.score;
}

// std::vector<float> xywh_xyxy(std::vector<float> bbox){

//     std::vector<float> res(4);
//     res[0] = bbox[0] - bbox[2]/2.f;
//     res[1] = bbox[1] - bbox[3]/2.f;
//     res[2] = bbox[0] + bbox[2]/2.f;
//     res[3] = bbox[1] + bbox[3]/2.f;
//     return res;
    
// }

float compute_iou(std::vector<float> lbox, std::vector<float> rbox){

    std::vector<float> interBox = {
        std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
        std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
        std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
        std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
    };

    if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
        return 0.0f;

    float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);  
    
}

void nms(std::vector<Detection>& detections){
    std::sort(detections.begin(), detections.end(), cmp);
    for(int i = 0; i < detections.size(); i++){
        for(int j = i+1; j < detections.size(); j++){
            if(compute_iou(detections[i].bbox, detections[j].bbox) > IOU_THRESH){
                detections.erase(detections.begin() + j);
                j--;
            }
        }
    } 
}


// void sigmoid(float* pdata, int length)
// {
// 	int i = 0; 
// 	for (i = 0; i < length; i++)
// 	{
// 		pdata[i] = 1.0 / (1 + std::expf(-pdata[i]));
// 	}
// }


cv::Rect get_rect(std::vector<float> bbox) {
    int l, r, t, b;

    l = bbox[0] - bbox[2]/2.f;
    t = bbox[1] - bbox[3]/2.f ;
    r = bbox[0] + bbox[2]/2.f;
    b = bbox[1] + bbox[3]/2.f;
    return cv::Rect(l, t, r-l, b-t);
}


cv::Rect get_rect(cv::Mat& img, std::vector<float> bbox) {
    int l, r, t, b;
    float r_w = INPUT_W / (img.cols * 1.0);
    float r_h = INPUT_H / (img.rows * 1.0);
    if (r_h > r_w) {
        l = bbox[0] - bbox[2]/2.f;
        r = bbox[0] + bbox[2]/2.f;
        t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
        l = l / r_w;
        r = r / r_w;
        t = t / r_w;
        b = b / r_w;
    } else {
        l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
        t = bbox[1] - bbox[3]/2.f;
        b = bbox[1] + bbox[3]/2.f;
        l = l / r_h;
        r = r / r_h;
        t = t / r_h;
        b = b / r_h;
    }
    return cv::Rect(l, t, r-l, b-t);
}

cv::Mat preprocess_img(cv::Mat& img, int input_w, int input_h) {
    int w, h, x, y;
    float r_w = input_w / (img.cols*1.0);
    float r_h = input_h / (img.rows*1.0);
    if (r_h > r_w) {
        w = input_w;
        h = r_w * img.rows;
        x = 0;
        y = (input_h - h) / 2;
    } else {
        w = r_h * img.cols;
        h = input_h;
        x = (input_w - w) / 2;
        y = 0;
    }
    cv::Mat re(h, w, CV_8UC3);
    cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    cv::Mat out(input_h, input_w, CV_8UC3, cv::Scalar(128, 128, 128));
    re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
    return out;
}



int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

// Creat the engine using only the API and not any parser.
ICudaEngine* createEngine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt)
{
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = builder->createNetworkV2(explicitBatch);
    auto parser = nvonnxparser::createParser(*network, gLogger.getTRTLogger());
    std::string onnx_file = "../yolov5s.onnx";
    if (!parser->parseFromFile(onnx_file.c_str(), static_cast<int>(gLogger.getReportableSeverity()))) {
        // gLogError << "Failure while parsing ONNX file" << std::endl;
        std::cout << "Failure while parsing ONNX file" << std::endl;
    }
    
    std::cout << "start building engine" << std::endl;
    
    // Build engine
    builder->setMaxBatchSize(maxBatchSize);
    config->setMaxWorkspaceSize(16 << 20);

#if defined(USE_FP16)
    config->setFlag(BuilderFlag::kFP16);

#elif defined(USE_INT8)
    std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
    assert(builder->platformHasFastInt8());
    config->setFlag(BuilderFlag::kINT8);
    Int8EntropyCalibrator2 *calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "../../coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
    config->setInt8Calibrator(calibrator);

#endif
    ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
    // Don't need the network any mor
    network->destroy();
    parser->destroy();
    return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream)
{
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = createEngine(maxBatchSize, builder, config, DataType::kFLOAT);
    assert(engine != nullptr);

    // Serialize the engine
    (*modelStream) = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();
}

void doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();

    // Pointers to input and output device buffers to pass to engine.
    // Engine requires exactly IEngine::getNbBindings() number of buffers.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    const int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME);
    const int outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);


    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], batchSize * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float)));

    // Create stream
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
    context.enqueueV2(buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

int main(int argc, char** argv)
{
    if (argc != 2 && argc !=3) {
        std::cerr << "arguments not right!" << std::endl;
        std::cerr << "./alexnet -s   // serialize model to plan file" << std::endl;
        std::cerr << "./yolov3 -d   image_dir // deserialize plan file and run inference" << std::endl;
        return -1;
    }

    // create a model using the API directly and serialize it to a stream
    char *trtModelStream{nullptr};
    size_t size{0};

    if (std::string(argv[1]) == "-s") {
        IHostMemory* modelStream{nullptr};
        APIToModel(1, &modelStream);
        assert(modelStream != nullptr);

        std::ofstream p("yolov5s.trt", std::ios::binary);
        if (!p)
        {
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        return 1;
    } else if (std::string(argv[1]) == "-d") {
        std::ifstream file("yolov5s.trt", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream = new char[size];
            assert(trtModelStream);
            file.read(trtModelStream, size);
            file.close();
        }
    } else {
        return -1;
    }
    // Subtract mean from image
    float *data = new float[3 * INPUT_H * INPUT_W];
    float *prob = new float[OUTPUT_SIZE];

    // for (int i = 0; i < 3 * INPUT_H * INPUT_W; i++)
    //     data[i] = 1;

    IRuntime* runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);

    // Run inference

    std::vector<std::string> file_names;
    if (read_files_in_dir(argv[2], file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    
    int fcount = 0;
    for (auto f: file_names) {
        fcount++;
        std::cout << fcount << "  " << f << std::endl;


        auto start = std::chrono::system_clock::now();
        cv::Mat img = cv::imread(std::string(argv[2]) + "/" + f);
        if (img.empty()) continue;
        cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);

        
        for (int i = 0; i < INPUT_H * INPUT_W; i++) {
            data[i] = pr_img.at<cv::Vec3b>(i)[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = pr_img.at<cv::Vec3b>(i)[0] / 255.0;
        }

        doInference(*context, data, prob, 1);
        
        float* pprob = prob;
        std::vector<Detection> detections;

        for(int l = 0; l< 3; l++){
            int stride = STRIDES[l];
            int grid_size_w = INPUT_W / stride;
            int grid_size_h = INPUT_H / stride;
            
            for(int a = 0; a < 3; a++){
                float anchor_w = ANCHORS[l][a*2];
                float anchor_h = ANCHORS[l][a*2 + 1];
                float* out = pprob + a*grid_size_w*grid_size_h*(NUM_CLASS + 5);
                for(int i = 0; i < grid_size_h; i++) {
                    for(int j = 0; j <grid_size_w; j++){
                        float* pout = out + i*grid_size_w*(NUM_CLASS + 5) + j*(NUM_CLASS + 5);
                        float conf_score = pout[4];
                        if(conf_score < CONF_THRESH){
                            continue;
                        }
                        float class_score = 0;
                        int class_id = 0;
                        for(int c = 5; c< NUM_CLASS + 5; c++){
                            if(pout[c] > class_score){
                                class_score = pout[c];
                                class_id = c - 5;
                            }
                        }
                        class_score = class_score*conf_score;
                        if(class_score < CLASS_THRESH){
                            continue;
                        }

                        // std::cout<<pout[0]<<" " <<pout[1]<<" " <<pout[2]<<" " <<pout[3]<<" " << class_id<<" "<<class_score<<std::endl;

                        float x = (pout[0]*2.f - 0.5f +  j) * stride;
                        float y = (pout[1]*2.f - 0.5f +  i) * stride;
                        float w = std::pow(pout[2]*2.f, 2.f)*anchor_w;
                        float h = std::pow(pout[3]*2.f, 2.f)*anchor_h; 
                        // std::cout<<x<<" " <<y<<" " <<w<<" " <<h<<" " << class_id<<" "<<class_score<<std::endl;
                        
                        std::vector<float> bbox = {x,y,w,h};
                        Detection detection;
                        detection.index = class_id;
                        detection.score = class_score;
                        detection.bbox = bbox;
                        detections.push_back(detection);
                    }
                }
            }   
            pprob += 3*grid_size_w *grid_size_h*(NUM_CLASS + 5);
        }

        nms(detections);
        

        for(int i = 0; i < detections.size(); i++){
            cv::Rect r = get_rect(img, detections[i].bbox);
            cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
            cv::putText(img, std::to_string(detections[i].index) + std::to_string(detections[i].score), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
        }
        cv::imwrite("_" + f, img);
        auto end = std::chrono::system_clock::now();
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

        
    }

    // for (int i = 0; i < 2; i++) {
    //     auto start = std::chrono::system_clock::now();
    //     doInference(*context, data, prob, 1);
    //     auto end = std::chrono::system_clock::now();
    //     std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    // }

    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    // Print histogram of the output distribution
    // std::cout << "\nOutput:\n\n";
    // for (unsigned int i = 0; i < 85; i++)
    // {
    //     std::cout << prob[i] << ", ";
    //     // if (i % 10 == 0) std::cout << i / 10 << std::endl;
    // }
    // std::cout << std::endl;
    delete [] prob; 
    delete [] data;
    return 0;
}