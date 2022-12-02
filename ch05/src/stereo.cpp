#include <opencv2/opencv.hpp>
// 调用PCL进行可视化
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using namespace cv;
using namespace std;

#define LINE cout << __LINE__ << endl;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "Usage: ./stereo <left_img> <right_img>" << endl;
        return -1;
    }
    Mat img_l = imread(argv[1]);
    Mat img_r = imread(argv[2]);
    if (img_l.data == NULL || img_r.data == NULL)
    {
        cout << "No images" << endl;
        return -1;
    }
    
    // 内参
    double fx = 718.856, fy = 718.856, cx = 607.1928, cy = 185.2157;
    // 基线
    double b = 0.573;

    cv::Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 96, 9,
                                                  8 * 9 * 9, 32 * 9 * 9, 1,
                                                  63, 10, 
                                                  100, 2);
    // CV_WRAP static Ptr<StereoSGBM> create(int minDisparity = 0, int numDisparities = 16, int blockSize = 3,
    //                                     int P1 = 0, int P2 = 0, int disp12MaxDiff = 0,
    //                                     int preFilterCap = 0, int uniquenessRatio = 0,
    //                                     int speckleWindowSize = 0, int speckleRange = 0,
    //                                     int mode = StereoSGBM::MODE_SGBM);
    // minDisparity     #表示可能的最小视差值。通常为0，但有时校正算法会移动图像，所以参数值也要相应调整
    // numDisparities   #表示最大的视差值与最小的视差值之差，这个差值总是大于0。在当前的实现中，这个值必须要能被16整除
    // blockseize       #匹配的块大小。它必须是一个大于1的奇数。通常它应该在3..11之间
    // P1 = P1          #控制视差图平滑度的第一个参数
    // P2 = P2          #控制视差图平滑度的第二个参数，值越大，视差图越平滑。P1是邻近像素间视差值变化为1时的惩罚值
    //                  #p2是邻近像素间视差值变化大于1时的惩罚值。算法要求P2>P1,stereo_match.cpp样例中给出一些p1和p2的合理取值。
    // disp12MaxDiff    #表示在左右视图检查中最大允许的偏差（整数像素单位）。设为非正值将不做检查。
    // preFilterCap     #预过滤图像像素的截断值
    // uniquenessRatio  #表示由代价函数计算得到的最好（最小）结果值比第二好的值小多少（用百分比表示）才被认为是正确的。通常在5-15之间。
    // speckleWindowSize#表示平滑视差区域的最大窗口尺寸，以考虑噪声斑点或无效性。将它设为0就不会进行斑点过滤，否则应取50-200之间的某个值。
    // speckleRange     #指每个已连接部分的最大视差变化，如果进行斑点过滤，则该参数取正值，函数会自动乘以16、一般情况下取1或2就足够了。
    // mode             #搜索方向

    Mat disparity_sgbm, disparity;
    sgbm->compute(img_l, img_r, disparity_sgbm);
    disparity_sgbm.convertTo(disparity, CV_32F, 1.0 / 16.0f);
LINE
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pointcloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for (size_t v = 0; v < img_l.rows; v++)
    {
        for (size_t u = 0; u < img_l.rows; u++)
        {
            if (disparity.at<float>(v, u) <= 0.0 || disparity.at<float>(v, u) >= 96.0) 
                continue;
            // 根据双目模型计算 point 的位置
            pcl::PointXYZRGB point;
            double x = (u - cx) / fx;
            double y = (v - cy) / fy;
            double depth = fx * b / (disparity.at<float>(v, u));
            point.x = x * depth;
            point.y = y * depth;
            point.z = depth;
            point.rgb = img_l.at<uchar>(v, u);

            pointcloud->push_back(point);
        }
    }
LINE
    pcl::visualization::PCLVisualizer viewer("Title of Windows");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> color_of_point(pointcloud, 255, 0, 0);
    viewer.addPointCloud(pointcloud, color_of_point, "ID");
    viewer.spin();
    return 0;
}