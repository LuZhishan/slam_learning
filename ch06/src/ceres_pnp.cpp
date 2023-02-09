#ifndef MOTION_BA_
#define MOTION_BA_

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>

#endif
// pnp class used in ceres ba
struct pnpCeres
{
private:
    cv::Point2d pt2d_;
    cv::Point3d pt3d_;
    cv::Mat K_cam_;

public:
    // 构造函数，传入相机内参和待优化的点数据
    pnpCeres(cv::Point2d pt2d, cv::Point3d pt3d, cv::Mat K): pt3d_(pt3d), pt2d_(pt2d), K_cam_(K) {}

    template<typename T>
    bool operator()(const T* const pose6d, T* residual) const { // pose6d是待优化变量
        T proj_pt2d[3];
        T proj_pt3d[3];
        proj_pt3d[0] = T(pt3d_.x);
        proj_pt3d[1] = T(pt3d_.y);
        proj_pt3d[2] = T(pt3d_.z);
        ceres::AngleAxisRotatePoint(pose6d, proj_pt3d, proj_pt2d);
        proj_pt2d[0] += pose6d[3];
        proj_pt2d[1] += pose6d[4];
        proj_pt2d[2] += pose6d[5];
        proj_pt2d[0] /= proj_pt2d[2];
        proj_pt2d[1] /= proj_pt2d[2];
        proj_pt2d[2] = T(1.0);

        residual[0] = proj_pt2d[0] * K_cam_.at<double>(0,0) + K_cam_.at<double>(0,2) - T(pt2d_.x);
        residual[1] = proj_pt2d[1] * K_cam_.at<double>(1,1) + K_cam_.at<double>(1,2) - T(pt2d_.y);

        return true;
    }

    static ceres::CostFunction* CreateCostFunction(const cv::Point2d pt2d, const cv::Point3d pt3d, const cv::Mat K){
        return (new ceres::AutoDiffCostFunction<pnpCeres, 2, 6>(// 2是输出误差的维度，6是待优化的维度
            new pnpCeres(pt2d, pt3d, K)));
    }
};

void bundleAdjustmentCeres(
    std::vector<cv::Point2d> pt2ds, 
    std::vector<cv::Point3d> pt3ds, 
    cv::Mat K, 
    double* init_pose6d)
{
    // create ceres problem for motion BA
    ceres::Problem problem;
    for (size_t i = 0; i < pt2ds.size(); i++)
    {
        ceres::LossFunction* loss = new ceres::HuberLoss(0.5);  // 核函数
        problem.AddResidualBlock(pnpCeres::CreateCostFunction(pt2ds[i], pt3ds[i], K), loss, init_pose6d);
    }
   
    // set ceres solver
    ceres::Solver::Options options;
    ceres::Solver::Summary summary;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations = 30;

    // solve the BA problem
    ceres::Solve(options, &problem, &summary);
    std::cout<< summary.BriefReport() <<std::endl;  
    
}