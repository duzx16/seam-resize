#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "seam.h"

using namespace cv;

int main()
{
    cv::Mat img = cv::imread("/Users/Duzx/Downloads/6.jpg");
    /*cv::Mat mask_img = cv::imread("/Users/Duzx/Downloads/content_mask.jpg");
    cv::Mat mask(mask_img.rows, mask_img.cols, CV_64F);
    for (int i = 0; i < mask_img.rows; ++i)
    {
        for (int j = 0; j < mask_img.cols; ++j)
        {
            if (mask_img.at<Vec3b>(i, j)[0] <= 10 and mask_img.at<Vec3b>(i, j)[1] <= 10 and mask_img.at<Vec3b>(i, j)[2] >= 240)
                mask.at<double>(i, j) = 1e20;
            else
                mask.at<double>(i, j) = 1.0;
        }
    }*/
    resize_img_vertical(img, 0.8, true);
    if (true)
    {
        // 进行90度旋转
        cv::transpose(img, img);
        cv::flip(img, img, 1);
        //cv::transpose(mask, mask);
        //cv::flip(mask, mask, 1);
        // 再次resize
        resize_img_vertical(img, 0.8, true);
        // 转回来
        cv::transpose(img, img);
        cv::flip(img, img, 0);
        //cv::transpose(mask, mask);
        //cv::flip(mask, mask, 0);
    }

    cv::imwrite("6.jpg", img);
    cv::imshow("[img]", img);
    cv::waitKey(0);
    return 0;
}