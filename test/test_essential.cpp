// C++
#include <assert.h>

// OpenCV
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

  // find features, match and compute essential matrix

  cout << "Computing Essential Matrix ... " << endl;
  rmatcher.robustMatchEssentialMat(img1, img2, K, inliers1, inliers2, inlier_matches, E);
  
  cout << "Essential: " << endl << E << endl;

  // put keypoints inside vector

  std::vector<cv::Point2d> points1, points2;
  for (int i = 0; i < inliers1.size(); ++i)
  {
    points1.push_back(inliers1[i].pt);
    points2.push_back(inliers2[i].pt);
  }

  // Recover pose

  Mat R, t;
  
  double focal = K.at<double>(0,0);
  cv::Point2d pp = cv::Point2d(K.at<double>(0,2),K.at<double>(1,2));

  recoverPose(E, points1, points2, R, t, focal, pp);

  cout << "R: " << endl << R << endl;
  cout << "t: " << endl << t << endl;

  /// Visualize

  // Create a window
  viz::Viz3d myWindow("3D Coordinate Frame");

  /// Add coordinate axes
  myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

  /// Let's assume camera has the following properties
  Point3d cam_pos(3.0f,3.0f,3.0f), cam_focal_point(3.0f,3.0f,2.0f), cam_y_dir(-1.0f,0.0f,0.0f);

  /// We can get the pose of the cam using makeCameraPose
  Affine3d cam_pose = viz::makeCameraPose(cam_pos, cam_focal_point, cam_y_dir);

  viz::WCameraPosition cpw(0.5); // Coordinate axes
  viz::WCameraPosition cpw_frustum(Vec2d(0.889484, 0.523599)); // Camera frustum
  myWindow.showWidget("CPW", cpw, cam_pose);
  myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, cam_pose);

  /// Construct pose
  Affine3d new_cam_pose(R, t);

  /// Display widget (update if already displayed)
  myWindow.showWidget("CPW", cpw, new_cam_pose);
  myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, new_cam_pose);

  /// Event loop for 1 millisecond
  myWindow.spin();

  return 0;
}
