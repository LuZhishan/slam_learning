#include <iostream>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main()
{
    // 旋转
    Matrix3d rotation_matrix = Matrix3d::Identity();        // 单位阵
    AngleAxisd rotation_vector(M_PI_4, Vector3d(0, 0, 1));  // 沿Z轴旋转45°
    cout.precision(3); // 设置输出精度，保留三位小数
    rotation_matrix = rotation_vector.matrix();              // 旋转向量转旋转矩阵
    rotation_matrix = rotation_vector.toRotationMatrix();

    // 旋转的坐标变换
    Vector3d v1(1, 0, 0);
    Vector3d v2 = rotation_vector * v1;
    Vector3d v3 = rotation_matrix * v1;

    // 欧拉角
    Vector3d euler_angle = rotation_matrix.eulerAngles(2, 1, 0);// yaw-pitch-roll顺序
    // 四元数
    Quaterniond q;                      // 默认w在前 wxyz
    q.coeffs();                         // coeffs的顺序是xyzw
    q = Quaterniond(rotation_vector);
    q = Quaterniond(rotation_matrix);

    // 变换矩阵Rt
    Isometry3d Rt = Isometry3d::Identity();     // 虽然称为3d，实质上是4＊4的矩阵
    Rt.translate(Vector3d(1,2,3));              // 平移部分t
    Rt.rotate(rotation_vector);                 // 旋转部分R————一定先平移后旋转

    // 旋转平移的坐标变换
    Vector3d v4 = Rt * v1;

    //对于仿射和射影变换，使用 Eigen::Affine3d 和 Eigen::Projective3d

    return 0;
}