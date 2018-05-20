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
        seam[n].push_back(min_index);
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
            seam[n].push_back(min_index);
        }
        std::reverse(std::begin(seam[n]), std::end(seam[n]));
        if(count > 1)
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
                output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j);
            } else
            {
                output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j + 1);
            }
        }
    }
}

void add_seam_vertical(Mat &img, Mat &output, const std::vector<std::vector<int>> &seam)
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
                if (j - count == img.cols)
                    output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j - count - 1);
                else
                {
                    Vec3b l = img.at<Vec3b>(i, j - 1 - count), r = img.at<Vec3b>(i, j - count);
                    output.at<Vec3b>(i, j) = Vec3b((l[0] + r[0]) / 2, (l[1] + r[1]) / 2, (l[2] + r[2]) / 2);
                }
                count += 1;
            } else
            {
                output.at<Vec3b>(i, j) = img.at<Vec3b>(i, j - count);
            }
        }
    }
}

void resize_img_vertical(Mat &img, double ratio)
{

    int new_cols = int(img.cols * ratio);
    if (new_cols < img.cols)
    {
        for (int i = 0; i < img.cols - new_cols; ++i)
        {
            Mat energy_mat;
            compute_energy(img, energy_mat);
            std::vector<std::vector<int>> seam;
            find_vertical_seam(energy_mat, seam, 1);
            Mat output(img.rows, img.cols - 1, CV_8UC3);
            remove_seam_vertical(img, output, seam[0]);
            img = output;
            std::cout << i << "\n";
        }
    } else
    {
        Mat energy_mat;
        compute_energy(img, energy_mat);
        std::vector<std::vector<int>> seam;
        find_vertical_seam(energy_mat, seam, new_cols - img.cols);
        Mat output(img.rows, new_cols, CV_8UC3);
        add_seam_vertical(img, output, seam);
        img = output;
    }
}

void resize_img_vertical(Mat &img, double ratio, Mat &mast_mat)
{

    int new_cols = int(img.cols * ratio);
    if (new_cols < img.cols)
    {
        for (int i = 0; i < img.cols - new_cols; ++i)
        {
            Mat energy_mat;
            compute_energy(img, energy_mat);
            std::vector<std::vector<int>> seam;
            find_vertical_seam(energy_mat, seam, 1);
            Mat output(img.rows, img.cols - 1, CV_8UC3);
            remove_seam_vertical(img, output, seam[0]);
            img = output;
            std::cout << i << "\n";
        }
    } else
    {
        Mat energy_mat;
        compute_energy(img, energy_mat);
        std::vector<std::vector<int>> seam;
        find_vertical_seam(energy_mat, seam, new_cols - img.cols);
        Mat output(img.rows, new_cols, CV_8UC3);
        add_seam_vertical(img, output, seam);
        img = output;
    }
}

