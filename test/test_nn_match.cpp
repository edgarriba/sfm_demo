// C++
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>

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

  Ptr<AKAZE> akaze = AKAZE::create();
  
  Ptr<DescriptorMatcher> matcher =
    DescriptorMatcher::create("BruteForce-Hamming");

  RobustMatcher rmatcher(akaze, matcher);
  rmatcher.setNNMatchRatio(nn_match_ratio);
  rmatcher.setRansacThreshold(ransac_thresh);

  cv::Mat desc1, desc2;
  std::vector<cv::KeyPoint> kpts1, kpts2;
  std::vector<std::vector<cv::DMatch> > nn_matches;

  // match features
  rmatcher.nnMatch(img1, kpts1, desc1, img2, kpts2, desc2, nn_matches);

  // Create & Open Window
  namedWindow("2D Coordinate Frame", WINDOW_KEEPRATIO);
  
  Mat res;
  drawMatches(img1, kpts1, img2, kpts2, nn_matches, res);
  imshow("2D Coordinate Frame", res);
  waitKey(0);
  

  // Close and Destroy Window
  destroyWindow("2D Coordinate Frame");

}
