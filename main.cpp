#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "seam.h"

int main()
{
    cv::Mat img = cv::imread("/Users/Duzx/Downloads/face.jpg");
    resize_img_vertical(img, 0.7);
    if(true)
    {
        // 进行90度旋转
        cv::transpose(img, img);
        cv::flip(img, img, 1);
        // 再次resize
        resize_img_vertical(img, 0.7);
        // 转回来
        cv::transpose(img, img);
        cv::flip(img, img, 0);
    }

    cv::imwrite("6.jpg", img);
    cv::imshow("[img]", img);
    cv::waitKey(0);
    return 0;
}