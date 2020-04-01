#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include "seam.h"

using namespace cv;

int main(int argc, char* argv[])
{
    assert(argc == 3);
    cv::Mat img = cv::imread(argv[1]), seam_img;
    cv::Mat mask(img.rows, img.cols, CV_64F, 0.0);
    cv::Mat mask_img = cv::imread(argv[2]);
    for (int i = 0; i < mask_img.rows; ++i)
    {
        for (int j = 0; j < mask_img.cols; ++j)
        {
            if (mask_img.at<Vec3b>(i, j)[0] <= 3 and mask_img.at<Vec3b>(i, j)[1] <= 3 and
                mask_img.at<Vec3b>(i, j)[2] >= 250)
                mask.at<double>(i, j) = -1e50;
            if (mask_img.at<Vec3b>(i, j)[1] <= 3 and mask_img.at<Vec3b>(i, j)[2] <= 3 and
                mask_img.at<Vec3b>(i, j)[0] >= 250)
                mask.at<double>(i, j) = 1e50;
        }
    }
    shrink_img(img, seam_img, 0.3, 0.0, mask);
    cv::imwrite("output.jpg", img);
    cv::imwrite("seam.jpg", seam_img);
    cv::imshow("[img]", img);
    cv::waitKey(0);
    return 0;
}