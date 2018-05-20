#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "seam.h"

using namespace cv;

int main()
{
    cv::Mat img = cv::imread("/Users/Duzx/Downloads/face.jpg"), seam_img;
    cv::Mat mask(img.rows, img.cols, CV_64F, 1.0);
    cv::Mat mask_img = cv::imread("/Users/Duzx/Downloads/content_mask.jpg");
    for (int i = 0; i < mask_img.rows; ++i)
    {
        for (int j = 0; j < mask_img.cols; ++j)
        {
            if (mask_img.at<Vec3b>(i, j)[0] == 0 and mask_img.at<Vec3b>(i, j)[1] == 0 and mask_img.at<Vec3b>(i, j)[2] == 255)
                mask.at<double>(i, j) = 1e20;
        }
    }
    shrink_img(img, seam_img, 0.2, 0.2, mask);
    cv::imwrite("face_cut.jpg", img);
    cv::imwrite("face_seam_cut.jpg", seam_img);
    cv::imshow("[img]", img);
    cv::waitKey(0);
    return 0;
}