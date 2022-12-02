#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace Eigen;

int main()
{
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    vector<double> x_data, y_data;
    for (size_t i = 0; i < N; i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar*x*x + br*x + cr) + rng.gaussian(w_sigma*w_sigma));
    }
    
    int iterations = 100;
    double cast = 0, lastcast = 0;
    for (size_t i = 0; i < N; i++)
    {
        double error = y_data[i] - exp(ae*x_data[i]*x_data[i] + be*x_data[i] + ce);
        Vector3d J;// 雅可比矩阵
        J[0] = 
    }
    





}