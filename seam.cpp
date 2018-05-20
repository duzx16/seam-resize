#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <iostream>
#include "seam.h"

using namespace cv;

template<typename T>
T my_min(const T &a, const T &b, const T &c)
{
    return a < b ? (a < c ? a : c) : (b < c ? b : c);
}

template<typename T>
T my_min(const T &a, const T &b)
{
    return a < b ? a : b;
}

void transpose(Mat &img, bool cw)
{
    if (cw)
    {
        cv::transpose(img, img);
        cv::flip(img, img, 1);
    } else
    {
        cv::transpose(img, img);
        cv::flip(img, img, 0);
    }
}

void compute_energy(cv::Mat &img, cv::Mat &out)
{
    cv::Mat dx, dy;
    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);
    cv::Sobel(gray, dx, CV_64F, 1, 0);
    cv::Sobel(gray, dy, CV_64F, 0, 1);
    //cv::Laplacian(gray, out, CV_64F);
    cv::magnitude(dx, dy, out);
}

void find_vertical_seam(cv::Mat &energy_mat, std::vector<std::vector<int>> &seam, int count)
{
    int base = seam.size();
    for (int n = 0; n < count; ++n)
    {
        int h = energy_mat.rows, w = energy_mat.cols;
        cv::Mat cost_mat(h, w, CV_64F, 0.0);
        for (int j = 0; j < w; ++j)
        {
            cost_mat.at<double>(0, j) = energy_mat.at<double>(0, j);
        }
        for (int i = 1; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                cost_mat.at<double>(i, j) = energy_mat.at<double>(i, j);
                if (j == 0)
                {
                    cost_mat.at<double>(i, j) += my_min(cost_mat.at<double>(i - 1, j),
                                                        cost_mat.at<double>(i - 1, j + 1));
                } else if (j == w - 1)
                {
                    cost_mat.at<double>(i, j) += my_min(cost_mat.at<double>(i - 1, j),
                                                        cost_mat.at<double>(i - 1, j - 1));
                } else
                {
                    cost_mat.at<double>(i, j) += my_min(cost_mat.at<double>(i - 1, j - 1),
                                                        cost_mat.at<double>(i - 1, j),
                                                        cost_mat.at<double>(i - 1, j + 1));
                }
            }
        }
        seam.emplace_back();
        double min_value = 1e200;
        int min_index = 0;
        for (int j = 0; j < w; ++j)
        {
            if (cost_mat.at<double>(h - 1, j) < min_value)
            {
                min_value = cost_mat.at<double>(h - 1, j);
                min_index = j;
            }
        }
        seam[base + n].push_back(min_index);
        energy_mat.at<double>(h - 1, min_index) = 1e20;
        for (int i = h - 2; i >= 0; --i)
        {
            if (min_index == 0)
            {
                if (cost_mat.at<double>(i, min_index + 1) < cost_mat.at<double>(i, min_index))
                {
                    min_index = min_index + 1;
                }
            } else if (min_index == w - 1)
            {
                if (cost_mat.at<double>(i, min_index - 1) < cost_mat.at<double>(i, min_index))
                {
                    min_index = min_index - 1;
                }

            } else
            {
                if (cost_mat.at<double>(i, min_index - 1) < cost_mat.at<double>(i, min_index + 1))
                {
                    if (cost_mat.at<double>(i, min_index - 1) < cost_mat.at<double>(i, min_index))
                    {
                        min_index = min_index - 1;
                    }
                } else
                {
                    if (cost_mat.at<double>(i, min_index + 1) < cost_mat.at<double>(i, min_index))
                    {
                        min_index = min_index + 1;
                    }
                }
            }
            energy_mat.at<double>(i, min_index) = 1e20;
            seam[base + n].push_back(min_index);
        }
        std::reverse(std::begin(seam[base + n]), std::end(seam[base + n]));
        if (count > 1)
        {
            std::cout << n << "\n";
        }
    }
}

void remove_seam_vertical(Mat &img, Mat &output, const std::vector<int> &seam)
{
    for (int i = 0; i < img.rows; ++i)
    {
        for (int j = 0; j < img.cols - 1; ++j)
        {
            if (j < seam[i])
            {
                if (img.depth() == CV_64F)
                {
                    output.at<double>(i, j) = img.at<double>(i, j);
                } else
                {
                    output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
                }
            } else
            {
                if (img.depth() == CV_64F)
                {
                    output.at<double>(i, j) = img.at<double>(i, j);
                } else
                {
                    output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j + 1);
                }
            }
        }
    }
}

void add_seam_vertical(Mat &img, Mat &output, const std::vector<std::vector<int>> &seam, bool show_seam)
{
    std::vector<std::vector<int>> new_seam;
    for (int i = 0; i < seam[0].size(); ++i)
    {
        new_seam.emplace_back();
        for (int j = 0; j < seam.size(); ++j)
        {
            new_seam[i].push_back(seam[j][i]);
        }
        new_seam[i].push_back(100000);
    }
    for (auto &it:new_seam)
    {
        std::sort(it.begin(), it.end());
    }
    for (int i = 0; i < img.rows; ++i)
    {
        int count = 0;
        for (int j = 0; j < img.cols + seam.size(); ++j)
        {
            if (j - count == new_seam[i][count] + 1)
            {
                if (show_seam)
                {
                    output.at<Vec3b>(i, j) = Vec3b(0, 0, 255);
                } else
                {
                    if (j - count == img.cols)
                        if (img.depth() == CV_64F)
                        {
                            output.at<double>(i, j) = img.at<double>(i, j - count - 1);
                        } else
                        {
                            output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j - count - 1);
                        }
                    else
                    {
                        if (img.depth() == CV_64F)
                        {
                            output.at<double>(i, j) =
                                    (img.at<double>(i, j - 1 - count) + img.at<double>(i, j - count)) / 2;
                        } else
                        {
                            Vec3b l = img.at<Vec3b>(i, j - 1 - count), r = img.at<Vec3b>(i, j - count);
                            output.at<Vec3b>(i, j) = Vec3b((l[0] + r[0]) / 2, (l[1] + r[1]) / 2, (l[2] + r[2]) / 2);

                        }
                    }
                }
                count += 1;
            } else
            {
                if (img.depth() == CV_64F)
                {
                    output.at<double>(i, j) = img.at<double>(i, j - count);
                } else
                {
                    output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j - count);
                }
            }
        }
    }
}


void shrink_img_vertical(Mat &img, int new_cols, Mat &mask_mat, std::vector<std::vector<int>> &seams)
{
    seams.clear();
    int n = img.cols - new_cols;
    if (new_cols < img.cols)
    {
        for (int i = 0; i < n; ++i)
        {
            Mat energy_mat;
            compute_energy(img, energy_mat);
            for (int x = 0; x < energy_mat.rows; x++)
            {
                for (int y = 0; y < energy_mat.cols; ++y)
                {
                    double weight = mask_mat.at<double>(x, y);
                    energy_mat.at<double>(x, y) = energy_mat.at<double>(x, y) * weight;
                }
            }
            find_vertical_seam(energy_mat, seams, 1);
            Mat output(img.rows, img.cols - 1, CV_8UC3), output_mask(img.rows, img.cols - 1, CV_64F);
            remove_seam_vertical(img, output, seams[i]);
            remove_seam_vertical(mask_mat, output_mask, seams[i]);
            img = output;
            mask_mat = output_mask;
            std::cout << i << "\n";
        }
    }

}

void show_removed_seams(Mat &seam_img, std::vector<std::vector<int>> &seams)
{
    for (int i = seams.size() - 1; i >= 0; --i)
    {
        Mat seam_output(seam_img.rows, seam_img.cols + 1, CV_8UC3);
        std::vector<std::vector<int>> temp;
        temp.emplace_back();
        for (int j = 0; j < seams[i].size(); ++j)
        {
            temp[0].push_back(seams[i][j] - 1);
        }
        add_seam_vertical(seam_img, seam_output, temp, true);
        seam_img = seam_output;
        std::cout << i << "\n";
    }
}

void shrink_img(Mat &img, Mat &seam_img, double v_ratio, double h_ratio, Mat &mask_mat)
{
    int new_cols = -int(img.cols * v_ratio) + img.cols, new_rows = -int(img.rows * h_ratio) + img.rows;
    std::vector<std::vector<int>> v_seams, h_seams;
    if (new_cols < img.cols)
    {
        shrink_img_vertical(img, new_cols, mask_mat, v_seams);
    }
    if (new_rows < img.rows)
    {
        transpose(img, true);
        transpose(mask_mat, true);
        shrink_img_vertical(img, new_rows, mask_mat, h_seams);
        transpose(img, false);
        transpose(mask_mat, false);
    }
    img.copyTo(seam_img);
    if (h_seams.size() > 0)
    {
        transpose(seam_img, true);
        show_removed_seams(seam_img, h_seams);
        transpose(seam_img, false);

    }
    if (v_seams.size() > 0)
    {
        show_removed_seams(seam_img, v_seams);
    }
}

void expand_img_vertical(Mat &img, Mat &seam_img, int new_cols, Mat &mask_mat)
{
    Mat energy_mat;
    compute_energy(img, energy_mat);
    for (int x = 0; x < energy_mat.rows; x++)
    {
        for (int y = 0; y < energy_mat.cols; ++y)
        {
            double weight = mask_mat.at<double>(x, y);
            energy_mat.at<double>(x, y) = energy_mat.at<double>(x, y) * weight;
        }
    }
    std::vector<std::vector<int>> seam;
    find_vertical_seam(energy_mat, seam, new_cols - img.cols);
    Mat output(img.rows, new_cols, CV_8UC3);
    Mat seam_img_output(img.rows, new_cols, CV_8UC3);
    Mat mask_output(img.rows, new_cols, CV_64F);
    add_seam_vertical(img, output, seam, false);
    add_seam_vertical(mask_mat, mask_output, seam, false);
    add_seam_vertical(seam_img, seam_img_output, seam, true);
    img = output;
    seam_img = seam_img_output;
    mask_mat = mask_output;
}

void expand_img(Mat &img, Mat &seam_img, double v_ratio, double h_ratio, Mat &mask_mat)
{
    int new_cols = int(img.cols * v_ratio) + img.cols, new_rows = int(img.rows * h_ratio) + img.rows;
    img.copyTo(seam_img);
    if (new_cols > img.cols)
    {
        expand_img_vertical(img, seam_img, new_cols, mask_mat);
    }
    if (new_rows > img.rows)
    {
        transpose(img, true);
        transpose(mask_mat, true);
        transpose(seam_img, true);
        expand_img_vertical(img, seam_img, new_rows, mask_mat);
        transpose(img, false);
        transpose(mask_mat, false);
        transpose(seam_img, false);
    }
}

