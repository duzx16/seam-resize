#ifndef SEAM_RESIZE_SEAM_H
#define SEAM_RESIZE_SEAM_H

#include <opencv2/core.hpp>

void expand_img(cv::Mat &img, cv::Mat &seam_img, double v_ratio, double h_ratio, cv::Mat &mask_mat);
void shrink_img(cv::Mat &img, cv::Mat &seam_img, double v_ratio, double h_ratio, cv::Mat &mask_mat);
#endif //SEAM_RESIZE_SEAM_H
