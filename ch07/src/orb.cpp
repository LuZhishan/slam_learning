#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        cout << "Usage: ./orb <img1> <img2>" << endl;
        return -1;
    }
    Mat img1 = imread(argv[1]);
    Mat img2 = imread(argv[2]);
    if (img1.data == NULL || img2.data == NULL) 
    {
        cout << "No images" << endl;
        return -1;
    }
    
    Mat img1_kp, img2_kp;
    vector<KeyPoint> kp1, kp2;      // 存储关键点
    Mat descriptors1, descriptors2; // 存储描述子
    Ptr<FeatureDetector> detector = ORB::create();      // 创建关键点检测器
    Ptr<DescriptorExtractor> descriptor = ORB::create();// 创建描述子检测器
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming"); // 创建汉明距离匹配器

    // 提取角点
    detector->detect(img1, kp1);
    detector->detect(img2, kp2);
    drawKeypoints(img1, kp1, img1_kp);  // 绘制关键点
    drawKeypoints(img2, kp2, img2_kp);

    // 提取角点的描述子
    descriptor->compute(img1, kp1, descriptors1);
    descriptor->compute(img2, kp2, descriptors2);
    // 构建匹配关系
    vector<DMatch> matches;
    matcher->match(descriptors1, descriptors2, matches);
    // 筛选匹配，剔除错误匹配
    auto min_max = minmax_element(matches.begin(), matches.end(),
            [](DMatch &m1, DMatch &m2){return m1.distance < m2.distance;}); // 根据汉明距离进行排序
    double min = min_max.first->distance;
    double max = min_max.second->distance;
    std::vector<DMatch> good_matches;
    for (size_t i = 0; i < descriptors1.rows; i++)
    {
        if (matches[i].distance < std::max(2 * min, 30.0))
        {
            good_matches.push_back(matches[i]);
        }
    }
    Mat img_match;
    drawMatches(img1, kp1, img2, kp2, good_matches, img_match); // 绘制匹配

    imshow("kp1", img1_kp);
    imshow("kp2", img2_kp);
    imshow("good matches", img_match);
    waitKey(0);
    return 0;
}