// C++
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>

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

  Mat essentialMat;
  vector<KeyPoint> kpts1_inliers, kpts2_inliers;
  vector<DMatch> inlier_matches;

  // find features, match and get essential matrix
  rmatcher.robustMatchEssentialMat(img1, img2, kpts1_inliers, kpts2_inliers, inlier_matches, essentialMat);


  // Create & Open Window
  namedWindow("2D Coordinate Frame", WINDOW_KEEPRATIO);


  Mat res;
  drawMatches(img1, kpts1_inliers, img2, kpts2_inliers, inlier_matches, res, Scalar(255,0,0), Scalar(255,0,0));
  imshow("2D Coordinate Frame", res);
  waitKey(0);
  

  // Close and Destroy Window
  destroyWindow("2D Coordinate Frame");

}
