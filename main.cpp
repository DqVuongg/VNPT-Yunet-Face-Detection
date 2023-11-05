#include "net.h"
#include "simpleocv.h"

#include <stdio.h>
#include <vector>
#include <string>
#include <iostream>

const int top_k = 5000;
const float scale = 1.0;
struct face
{
    cv::Rect_<float> rect;
    float score;
};
static inline float intersection_area(const face &a, const face &b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}
static void qsort(std::vector<face> &faces, int left, int right)
{
    int i = left;
    int j = right;
    float p = faces[(left + right) / 2].score;
    while (i <= j)
    {
        while (faces[i].score > p)
        {
            i++;
        }
        while (faces[j].score < p)
        {
            j--;
        }
        if (i <= j)
        {
            std::swap(faces[i], faces[j]);
            i++;
            j--;
        }
    }

    if (left < j)
        qsort(faces, left, j);

    if (i < right)
        qsort(faces, i, right);
}
static void qsort(std::vector<face> &faces)
{
    if (faces.empty())
        return;
    qsort(faces, 0, faces.size() - 1);
}
static void nmsBox(const std::vector<face> &faces, std::vector<int> &picked, float nms_threshold)
{
    picked.clear();
    int n = faces.size();
    std::vector<float> area(n);
    for (int i = 0; i < n; i++)
    {
        area[i] = faces[i].rect.area();
    }
    for (int i = 0; i < n; i++)
    {
        const face &a = faces[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const face &b = faces[picked[j]];
            float inter = intersection_area(a, b);
            float uni = area[i] + area[picked[j]] - inter;
            if (inter / uni > nms_threshold)
            {
                keep = 0;
            }
        }
        if (keep == 1)
            picked.push_back(i);
    }
}
static void YunetDetect(const cv::Mat &bgr, std::vector<face> &faces)
{
    const float conf_threshold = 0.9f;
    const float nms_threshold = 0.2f;
    ncnn::Net net;
    net.opt.use_vulkan_compute = true;
    if (net.load_param("/home/vuong/Yunet/models/face_detection_yunet_2023mar-sim-opt.param")) /// home/vuong/Yunet/models/Yunet-sim-opt.param
        exit(-1);
    if (net.load_model("/home/vuong/Yunet/models/face_detection_yunet_2023mar-sim-opt.bin")) /// home/vuong/Yunet/models/Yunet-sim-opt.bin
        exit(-1);
    // if (net.load_param("yunet_n_320_320-sim-opt.param")) /// home/vuong/Yunet/models/Yunet-sim-opt.param
    //     exit(-1);
    // if (net.load_model("yunet_n_320_320-sim-opt.bin")) /// home/vuong/Yunet/models/Yunet-sim-opt.bin
    //     exit(-1);
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels(bgr.data, ncnn::Mat::PIXEL_BGR, bgr.cols, bgr.rows);
    int divisor = 32;
    int inputW = img_w;
    int inputH = img_h;
    int padW = ((inputW - 1) / divisor + 1) * divisor;
    int padH = ((inputH - 1) / divisor + 1) * divisor;
    printf("%d %d\n", inputW, inputH);
    ncnn::Extractor ex = net.create_extractor();

    ex.input("input", in);
    const std::vector<int> strides = {8, 16, 32};
    // std::vector<string> output_names = {"cls_8", "cls_16", "cls_32", "obj_8", "obj_16", "obj_32", "bbox_8", "bbox_16", "bbox_32"};
    std::vector<ncnn::Mat> output_blobs;
    ex.extract("cls_8", output_blobs[0]);
    ex.extract("cls_16", output_blobs[1]);
    ex.extract("cls_32", output_blobs[2]);
    ex.extract("obj_8", output_blobs[3]);
    ex.extract("obj_16", output_blobs[4]);
    ex.extract("obj_32", output_blobs[5]);
    ex.extract("bbox_8", output_blobs[6]);
    ex.extract("bbox_16", output_blobs[7]);
    ex.extract("bbox_32", output_blobs[8]);
    // for (int i = 0; i < output_names.size(); i++)
    // {
    //     ex.extract(output_names[i], output_blobs[i]);
    // }
    std::vector<face> faceproposal;
    for (size_t i; i < strides.size(); i++)
    {
        face face;
        ncnn::Mat cls = output_blobs[i];
        ncnn::Mat obj = output_blobs[i + strides.size()];
        ncnn::Mat bbox = output_blobs[i + strides.size() * 2];
        float *cls_v = (float *)cls.data;
        float *obj_v = (float *)obj.data;
        float *bbox_v = (float *)bbox.data;

        int cols = padH / strides[i];
        int rows = padW / strides[i];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                size_t idx = r * cols + c;
                // Get score
                float cls_score = cls_v[idx];
                float obj_score = obj_v[idx];
                if (obj_score < 0.f)
                {
                    obj_score = 0.f;
                }
                else if (obj_score > 1.f)
                {
                    obj_score = 1.f;
                }
                if (cls_score < 0.f)
                {
                    cls_score = 0.f;
                }
                else if (cls_score > 1.f)
                {
                    cls_score = 1.f;
                }
                float score = std::sqrt(cls_score * obj_score);
                float cx = (c + bbox_v[idx * 4 + 0]) * strides[i];
                float cy = (r + bbox_v[idx * 4 + 1]) * strides[i];
                float w = exp(bbox_v[idx * 4 + 2]) * strides[i];
                float h = exp(bbox_v[idx * 4 + 3]) * strides[i];
                float x1 = cx - w / 2.f;
                float y1 = cy - h / 2.f;
                if (score >= conf_threshold)
                {
                    face.score = score;
                    face.rect.x = x1;
                    face.rect.y = y1;
                    face.rect.width = w;
                    face.rect.height = h;
                    faceproposal.push_back(face);
                }
            }
        }
    }
    qsort(faceproposal);
    std::vector<int> picked;
    nmsBox(faceproposal, picked, nms_threshold);
    int count = picked.size();
    faces.resize(count);
    fprintf(stderr, "%d\n", count);

    // for (int i = 0; i < count; i++)
    // {
    //     faces[i] = faceproposal[picked[i]];
    //     float x0 = faces[i].rect.x;
    //     float y0 = faces[i].rect.y;
    //     float x1 = x0 + faces[i].rect.width;
    //     float y1 = y0 + faces[i].rect.height;

    //     x0 = std::max(std::min(x0, (float)img_w - 1), 0.f);
    //     y0 = std::max(std::min(y0, (float)img_h - 1), 0.f);
    //     x1 = std::max(std::min(x1, (float)img_w - 1), 0.f);
    //     y1 = std::max(std::min(y1, (float)img_h - 1), 0.f);
    //     // fprintf(stderr, "%.2f %.2f %.2f %.2f\n", x0, y0, x1, y1);
    //     faces[i].rect.x = x0;
    //     faces[i].rect.y = y0;
    //     faces[i].rect.width = x1 - x0;
    //     faces[i].rect.height = y1 - y0;
    // }
    // cv::Mat image = bgr.clone();
    // for (size_t i = 0; i < faces.size(); i++)
    // {
    //     const face &fc = faces[i];
    //     cv::rectangle(image, fc.rect, cv::Scalar(255, 0, 0), 2);
    // }
    // cv::imshow("image", image);
    // cv::waitKey(0);
}
int main()
{
    cv::Mat m = cv::imread("/home/vuong/Yunet/img/nasa.jpg", 1); /// home/vuong/Yunet/img/sample.jpg
    // cv::Mat m = cv::imread("img/nasa.jpg", 1);
    if (m.empty())
    {
        fprintf(stderr, "fail to load\n");
        return -1;
    }
    std::vector<face> faces;
    YunetDetect(m, faces);
}
