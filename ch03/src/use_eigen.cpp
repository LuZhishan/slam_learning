#include <iostream>
#include <Eigen/Core>
#include <Eigen/Dense> // 特征值特征向量所需的头文件

using namespace std;
using namespace Eigen;

int main()
{
    Matrix<float, 2, 3> matrix_23; // 创建一个2行3列的矩阵
    Matrix3d matrix_33d = Matrix3d::Zero(); //初始化为零
    Matrix3f matrix_33f = Matrix3f::Ones();
    MatrixXd matrix_x; // 未指定大小

    matrix_23 << 1, 2, 3, 4, 5, 6;
    cout << matrix_23 << endl;
    cout << matrix_23(1, 2) << endl; // 访问指定位置的数据

    // 不同数据类型不能直接相加，必须显式转换
    // Matrix3d sum = matrix_33d + matrix_33f;
    Matrix3d sum = matrix_33d + matrix_33f.cast<double>();

    Matrix3d matrix_33 = Matrix3d::Random();                    // 随机数矩阵
    cout << "random matrix: \n" << matrix_33 << endl;
    cout << "transpose: \n" << matrix_33.transpose() << endl;   // 转置
    cout << "sum: " << matrix_33.sum() << endl;                 // 各元素和
    cout << "trace: " << matrix_33.trace() << endl;             // 迹
    cout << "times 10: \n" << 10 * matrix_33 << endl;           // 数乘
    cout << "inverse: \n" << matrix_33.inverse() << endl;       // 逆
    cout << "det: " << matrix_33.determinant() << endl;         // 行列式

    // 求解特征值和特征向量
    SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
    cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
    cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

    // QR分解求逆 
    //matrix_NN * x = v_Nd
    Matrix3d matrix_NN = matrix_33.transpose() * matrix_33;
    Vector3d v_nd = Vector3d::Random();
    Vector3d x1 = matrix_NN.inverse() * v_nd;                   // 直接求逆
    Vector3d x2 = matrix_NN.colPivHouseholderQr().solve(v_nd);  // QR分解
    
    return 0;
}