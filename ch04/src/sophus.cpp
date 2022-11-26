#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>

using namespace std;
using namespace Eigen;

int main()
{
/*旋转**********************************/
    Matrix3d R = AngleAxisd(M_PI / 2, Vector3d(0, 0, 1)).toRotationMatrix();
    Quaterniond q(R);
    // 送旋转矩阵或者四元数构建李群SO3
    Sophus::SO3d SO3_R(R);
    Sophus::SO3d SO3_q(q);
    cout << "SO(3)" << SO3_R.matrix() << endl; // 以矩阵形式输出

    // 构建李代数
    Vector3d so3 = SO3_R.log();
    SO3_R = Sophus::SO3d::exp(so3);
    cout << "SO(3)" << SO3_R.matrix() << endl;
    // 反对称矩阵
    Eigen::Matrix3d mat_h = Sophus::SO3d::hat(so3); // 李代数到反对称矩阵
    Vector3d v_vee = Sophus::SO3d::vee(mat_h);      // 反对称矩阵到李代数


    // 扰动模型
    Vector3d update_so3(1e-4, 0, 0);    //假设更新量为这么多
    Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
    cout << "SO3 updated = \n" << SO3_updated.matrix() << endl;

/*旋转和平移*******************************/
    Vector3d t(1, 0, 0);
    Sophus::SE3d SE3_Rt(R, t);  // 从R和t构建李群SE3

    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    Vector6d se3 = SE3_Rt.log();
    cout << "se3 = " << se3.transpose() << endl;

    Vector6d update_se3; //更新量
    update_se3.setZero();
    update_se3(0, 0) = 1e-4;
    Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
    cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;



}