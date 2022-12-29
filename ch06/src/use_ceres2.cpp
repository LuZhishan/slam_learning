#include <ceres/ceres.h>

struct F1 
{
    template <typename T> bool operator()(const T* const x1,
                                            const T* const x2,
                                            T* residual) const 
    {
        // f1 = x1 + 10 * x2 + 4
        residual[0] = x1[0] + 10.0 * x2[0] + 4.0;
        return true;
    }
};
struct F2 
{
    template <typename T> bool operator()(const T* const x3,
                                            const T* const x4,
                                            T* residual) const 
    {
        // f2 = sqrt(5) (x3 - x4) + 3
        residual[0] = sqrt(5.0) * (x3[0] - x4[0]) + 3.0;
        return true;
    }
};
struct F3 
{
    template <typename T> bool operator()(const T* const x2,
                                            const T* const x3,
                                            T* residual) const 
    {
        // f3 = (x2 - 2 x3)^2 - 2
        residual[0] = (x2[0] - 2.0 * x3[0]) * (x2[0] - 2.0 * x3[0]) - 2.0;
        return true;
    }
};
struct F4 {
    template <typename T> bool operator()(const T* const x1,
                                            const T* const x4,
                                            T* residual) const 
    {
        // f4 = sqrt(10) (x1 - x4)^2 - 1
        residual[0] = sqrt(10.0) * (x1[0] - x4[0]) * (x1[0] - x4[0]) - 1.0;
        return true;
    }
};
 
 
int main(int argc, char** argv) 
{
    double x1 =  3.0;
    double x2 = -1.0;
    double x3 =  0.0;
    double x4 =  1.0;
    ceres::Problem problem;
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F1, 1, 1, 1>(new F1),
                            NULL,
                            &x1, &x2);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F2, 1, 1, 1>(new F2),
                            NULL,
                            &x3, &x4);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F3, 1, 1, 1>(new F3),
                            NULL,
                            &x2, &x3);
    problem.AddResidualBlock(new ceres::AutoDiffCostFunction<F4, 1, 1, 1>(new F4),
                            NULL,
                            &x1, &x4);
    ceres::Solver::Options options;
    
    options.max_num_iterations = 100;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    std::cout << "Initial x1 = " << x1
                << ", x2 = " << x2
                << ", x3 = " << x3
                << ", x4 = " << x4
                << "\n";
    // Run the solver!
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
    std::cout << "Final x1 = " << x1
                << ", x2 = " << x2
                << ", x3 = " << x3
                << ", x4 = " << x4
                << "\n";
    return 0;
}