// Sfm demo
#include <params.h>
#include <robust_matcher.h>

using namespace cv;
using namespace std;


/**  Main program  **/
int main(int argc, char *argv[])
{

  Mat img1 = imread(img1_path, IMREAD_GRAYSCALE);
  Mat img2 = imread(img2_path, IMREAD_GRAYSCALE);

  if( img2.empty() || img2.empty())
  {
    cout << "Could not posible to open images" << endl;
    return 0;
  }

  // setup features detector
  Ptr<AKAZE> akaze = AKAZE::create();
  
  // setup descriptors matcher
  Ptr<DescriptorMatcher> matcher =
    DescriptorMatcher::create("BruteForce-Hamming");

  // setup robust matcher object
  RobustMatcher rmatcher(akaze, matcher);
  rmatcher.setNNMatchRatio(nn_match_ratio);
  rmatcher.setRansacThreshold(ransac_thresh);

  Mat E, F;
  vector<KeyPoint> inliers1, inliers2;
  vector<DMatch> inlier_matches;

  // find features, match and get essential matrix

  cout << "Computing Essential Matrix ... " << endl;
  rmatcher.robustMatchEssentialMat(img1, img2, K, inliers1, inliers2, inlier_matches, E);
  
  cout << "Computing Fundamental Matrix ... " << endl;
  rmatcher.robustMatchFundamentalMat(img1, img2, inliers1, inliers2, inlier_matches, F);

  Mat R, t;
  std::vector<cv::Point2f> points1, points2;

  // put keypoints inside vector
  for (int i = 0; i < inliers1.size(); ++i)
  {
    points1.push_back(inliers1[i].pt);
    points2.push_back(inliers2[i].pt);
  }

  // Recover pose from essential matrix
  double focal = 1.0;
  Point2d pp = Point2d(0, 0);
  recoverPose(E, points1, points2, R, t);

  cout << "R: " << std::endl << R <<endl;
  cout << "t: " << std::endl << t <<endl;

  
  // Camera Matrix
  Mat K = (Mat_<double>(3, 3) << width*f/sx, 0, width/2,   // fx 0 cx
                               0, height*f/sy, height/2,   // 0 fy cx
                               0, 0, 1);                   // 0  0  1

    // compute essential from fundamental
    E = K.t() * F * K;
/*
    std::vector<cv::Point2f> points1, points2;

    // put keypoints inside vector
    for (int i = 0; i < inliers1.size(); ++i)
    {
      points1.push_back(inliers1[i].pt);
      points2.push_back(inliers2[i].pt);
    }

    cout << "E: " << std::endl << points1.size() <<endl;
    cout << "F: " << std::endl << points2.size() <<endl;

    // Recover pose from essential matrix
    cout << "E: " << std::endl << E <<endl;
    cout << "F: " << std::endl << F <<endl;

    double focal = f;
    cv::Point2d pp = cv::Point2d(width/2,height/2);
    recoverPose(E, points1, points2, R, t, focal, pp);
    //recoverPose(E, points1, points2, R, t);
*/
    


    SVD svd(E);
    Matx33d W(0,-1, 0,   //HZ 9.13
              1, 0, 0,
              0, 0, 1);
    Matx33d Winv(0,1,0,
                -1,0,0,
                 0,0,1);

    R = svd.u * Mat(W) * svd.vt; //HZ 9.19
    t = svd.u.col(2); //u3
    
    cout << "R_diff: " << std::endl << R <<endl;
    cout << "t_diff: " << std::endl << t <<endl;

  return 0;
}
