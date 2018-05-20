#ifndef SEAM_RESIZE_SEAM_H
#define SEAM_RESIZE_SEAM_H

#include <opencv2/core.hpp>
void resize_img_vertical(cv::Mat &img, double ratio, bool show_seam=false);
void resize_img_vertical(cv::Mat &img, double ratio, cv::Mat &mask_mat, bool show_seam=false);
#endif //SEAM_RESIZE_SEAM_H
