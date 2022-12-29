#include <ceres/ceres.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;

struct CURVE_FITTING_COST
{
    // 结构体的构造函数，输入x和y，赋值给_x和_y
    CURVE_FITTING_COST(double x, double y): _x(x), _y(y) {} 

    template<typename T>
    bool operator()(const T *const abc, T *residual) const  // 括号里面的是待优化变量abc，abc的维度由构造函数指定
    {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]);
        return true;
    }

    double _x, _y;
};




int main() 
{
    double ar = 1.0, br = 2.0, cr = 1.0;        // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;       // 估计参数值
    int N = 100;                                // 数据点
    double w_sigma = 1.0;                       // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    RNG rng;                                // OpenCV随机数产生器

    vector<double> x_data, y_data;              // 数据
    for (int i = 0; i < N; i++) 
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = {ae, be, ce};
    ceres::Problem problem;
    for (size_t i = 0; i < N; i++)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>( // 这里的1，3指定了“输出”、“输入”变量的维度（先输出后输入）
                new CURVE_FITTING_COST(x_data[i], y_data[i])),
            nullptr,// 核函数
            abc     // 待估计参数
        );
    }
    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    cout << summary.BriefReport() << endl;

    return 0;
}