#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    if (argc != 2)  
    {
        cout << "Usage: ./undistort <img>" << endl;
        return -1;
    }
    Mat img_in = imread(argv[1], 0);
    if (img_in.data == NULL)
    {
        cout << "No such img" << endl;
        return -1;
    }
    
    // 畸变参数
    double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
    // 内参
    double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

    Mat img_out = Mat(img_in.rows, img_in.cols, CV_8UC1);
    for (size_t v = 0; v < img_in.rows; v++)
    {
        for (size_t u = 0; u < img_in.cols; u++)
        {
            double x = (u -cx) / fx, y = (v -cy) / fy;
            double r = hypot(x, y);
            double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
            double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
            double u_distorted = fx * x_distorted + cx;
            double v_distorted = fy * y_distorted + cy;

            // 赋值 (最近邻插值)
            if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < img_in.cols && v_distorted < img_in.rows) 
            {
                img_out.at<uchar>(v, u) = img_in.at<uchar>((int) v_distorted, (int) u_distorted);
            } 
            else 
            {
                img_out.at<uchar>(v, u) = 0;
            }
        }
        
    }
    







}