#include <opencv2/opencv.hpp>

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

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) 
{
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1); // 相机内参

    vector<Point2f> points1;
    vector<Point2f> points2;
    for (int i = 0; i < (int) matches.size(); i++) 
    {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, FM_8POINT); // 计算基础矩阵
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, K);   // 计算本质矩阵
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    recoverPose(essential_matrix, points1, points2, K, R, t);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;
}

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
    vector<KeyPoint> kp1, kp2;
    vector<DMatch> good_matches;

    find_feature_matches(img1, img2, kp1, kp2, good_matches);

    Mat R, t;
    pose_estimation_2d2d(kp1, kp2, good_matches, R, t);



    return 0;
}