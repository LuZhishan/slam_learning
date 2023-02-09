#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>   // eigen必须在opencv前面
#include <sophus/se3.hpp>

#include <g2o/core/block_solver.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace cv;
using namespace std;

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) 
{
    Mat descriptors_1, descriptors_2;
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    vector<DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);
    auto min_max = minmax_element(match.begin(), match.end(),
            [](DMatch &m1, DMatch &m2){return m1.distance < m2.distance;}); // 根据汉明距离进行排序
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    for (int i = 0; i < descriptors_1.rows; i++) 
    {
        if (match[i].distance <= max(2 * min_dist, 30.0))
        {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) 
{
  return Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}


// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void setToOriginImpl() override {
        _estimate = Sophus::SE3d();
    }

    /// left multiplication on SE3
    virtual void oplusImpl(const double *update) override {
        Eigen::Matrix<double, 6, 1> update_eigen;
        update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
        _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> 
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

    virtual void computeError() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
        pos_pixel /= pos_pixel[2];
        _error = _measurement - pos_pixel.head<2>();
    }

    virtual void linearizeOplus() override {
        const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
        Sophus::SE3d T = v->estimate();
        Eigen::Vector3d pos_cam = T * _pos3d;
        double fx = _K(0, 0);
        double fy = _K(1, 1);
        double cx = _K(0, 2);
        double cy = _K(1, 2);
        double X = pos_cam[0];
        double Y = pos_cam[1];
        double Z = pos_cam[2];
        double Z2 = Z * Z;
        _jacobianOplusXi
        << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
        0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
    }

    virtual bool read(istream &in) override {}
    virtual bool write(ostream &out) const override {}

private:
    Eigen::Vector3d _pos3d;
    Eigen::Matrix3d _K;
};

void bundleAdjustmentG2O(
    const vector<Eigen::Vector3d> &points_3d,
    const vector<Eigen::Vector2d> &points_2d,
    const Mat &K,
    Sophus::SE3d &pose_in,
    Sophus::SE3d &pose_out) 
{
    // 构建图优化，先设定g2o
    // 这里的6，2是这样：6是待优化的变量的维度SE3的维度，2的误差的维度，
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // 线性求解器类型
    // 梯度下降方法，可以从GN, LM, DogLeg 中选
    auto solver = new g2o::OptimizationAlgorithmGaussNewton(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;   // 图模型
    optimizer.setAlgorithm(solver);   // 设置求解器
    optimizer.setVerbose(true);       // 打开调试输出

    // vertex
    VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(pose_in);
    optimizer.addVertex(vertex_pose);

    // K
    Eigen::Matrix3d K_eigen;
    K_eigen <<
        K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
        K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
        K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

    // edges
    int index = 1;
    for (size_t i = 0; i < points_2d.size(); ++i) {
        auto p2d = points_2d[i];
        auto p3d = points_3d[i];
        EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
        edge->setId(index);
        edge->setVertex(0, vertex_pose);
        edge->setMeasurement(p2d);
        edge->setInformation(Eigen::Matrix2d::Identity());
        optimizer.addEdge(edge);
        index++;
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    optimizer.setVerbose(true);
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optimization costs time: " << time_used.count() << " seconds." << endl;
    cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
    pose_out = vertex_pose->estimate();
}


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



int main(int argc, char **argv)
{
    if (argc != 4)
    {
        cout << "Usage: pose_estimation_3d2d img1 img2 depth1" << endl;
        return -1;
    }
    Mat img_1 = imread(argv[1]);
    Mat img_2 = imread(argv[2]);
    Mat depth_1 = imread(argv[3], -1); // 深度图为16位无符号数，单通道图像
    if (img_1.data == NULL || img_2.data == NULL || depth_1.data == NULL)
    {
        cout << "No such images" << endl;
        return -1;
    }
    vector<KeyPoint> kp1, kp2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, kp1, kp2, matches);

    vector<Point3d> pts_3d;
    vector<Point2d> pts_2d;
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    for (DMatch m:matches) 
    {
        ushort d = depth_1.ptr<unsigned short>(int(kp1[m.queryIdx].pt.y))[int(kp1[m.queryIdx].pt.x)];
        if (d == 0)   // bad depth
        continue;
        float dd = d / 5000.0;
        Point2d p1 = pixel2cam(kp1[m.queryIdx].pt, K);
        pts_3d.push_back(Point3f(p1.x * dd, p1.y * dd, dd));
        pts_2d.push_back(kp2[m.trainIdx].pt);
    }

    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false);
    Mat R;
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵
    cout << R << endl;

    Eigen::MatrixXd R_e(R.rows, R.cols);// 根据Mat的尺寸构建Matrix
    Eigen::MatrixXd t_e(t.rows, t.cols);
    cv2eigen(R, R_e);                   // 将Mat转换成Matrix
    cv2eigen(t, t_e);

    vector<Eigen::Vector3d> pts_3d_eigen;
    vector<Eigen::Vector2d> pts_2d_eigen;
    for (size_t i = 0; i < pts_3d.size(); ++i) 
    {
        pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
        pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
    }

    Sophus::SE3d pose_pnp(R_e, t_e);
    Eigen::AngleAxisd r0(0.0, Eigen::Vector3d(0, 0, 1)); Eigen::Vector3d t0(0,0,0);
    Sophus::SE3d pose_0(r0.toRotationMatrix(), t0);
    Sophus::SE3d pose_g2o;
    // 无论从谁开始，精度相当，当然也有可能是因为Rt本来就很小
    // 而且即使不进行优化，只有pnp精度也差不多。
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_0, pose_g2o);   // 从零开始所需时间0.466712ms
    bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_pnp, pose_g2o); // 从pnp开始所需时间0.223972ms
    double pose[6] = {0,0,0,0,0,0}; // 这个pose既是输入的初始值，也是最后输出值
    bundleAdjustmentCeres(pts_2d, pts_3d, K, pose);
    for (size_t i = 0; i < 6; i++)
    {
        cout << pose[i] << endl;
    }

    return 0;
}