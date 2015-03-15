// C++
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/viz.hpp>

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

  Mat E;
  vector<KeyPoint> inliers1, inliers2;
  vector<DMatch> inlier_matches;

  // find features, match and get essential matrix
  rmatcher.robustMatchEssentialMat(img1, img2, inliers1, inliers2, inlier_matches, E);

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


  /// Create a window
  viz::Viz3d myWindow("3D Coordinate Frame");

  /// Add coordinate axes
  myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

  // Load mesh
  /*iz::Mesh box_mesh;
  box_mesh.load(mesh_path);
  cout << mesh_path << endl;
  cout << box_mesh.cloud << endl;
  viz::WCloud cloud_widget(box_mesh.cloud, viz::Color::green());
  myWindow.showWidget("box", cloud_widget);*/

  /// Construct a cube widget
  viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
  cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);

  /// Construct pose
  Affine3d pose(R, t);

  /// Display widget (update if already displayed)
  myWindow.showWidget("Cube Widget", cube_widget, pose);

  /// Wait for key q
  myWindow.spin();

  return 0;
}
