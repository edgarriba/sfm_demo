// C++
#include <iostream>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>

// Sfm demo
#include <params.h>
#include <robust_matcher.h>

using namespace cv;
using namespace std;


/**  Main program  **/
int main(int argc, char *argv[])
{
  VideoCapture capture;            // instantiate VideoCapture
  capture.open(video1_path);       // open a recorded video

  if(!capture.isOpened())          // check if we succeeded
  {
    cout << "Could not open the camera device" << endl;
    return -1;
  }

  Ptr<AKAZE> akaze = AKAZE::create();
  
  Ptr<DescriptorMatcher> matcher =
    DescriptorMatcher::create("BruteForce-Hamming");
//    DescriptorMatcher::create("FlannBased");

  RobustMatcher rmatcher(akaze, matcher);
  rmatcher.setNNMatchRatio(nn_match_ratio);
  rmatcher.setRansacThreshold(ransac_thresh);

  Mat current_frame, previous_frame;

  Mat E, F;
  Mat R, t, C, T;
  vector<KeyPoint> inliers1, inliers2;
  vector<DMatch> inlier_matches;

  Mat desc1, desc2;
  vector<KeyPoint> kpts1, kpts2;
  vector<vector<DMatch> > nn_matches;

  /// Create a window
  viz::Viz3d myWindow("3D Coordinate Frame");
  cv::namedWindow("2D Coordinate Frame");

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

  // initial frame
  capture >> current_frame;
  C = Mat::eye(4, 4, CV_64F);

  resize(current_frame, current_frame, Size(), 0.25, 0.25, INTER_LINEAR);

  int counter = 1;

  for( ;; )
  {
    // set previous data
    previous_frame = current_frame;

    capture >> current_frame;

    resize(current_frame, current_frame, Size(), 0.25, 0.25, INTER_LINEAR);

    cout << "Computing frame " << counter << std::endl;

    // find features, match and get essential matrix
    //rmatcher.robustMatchEssentialMat(current_frame, previous_frame, inliers1, inliers2, inlier_matches, E);
    
    // find features, match and get fundamental matrix
    rmatcher.robustMatchFundamentalMat(current_frame, previous_frame, inliers1, inliers2, inlier_matches, F);

    // default params
    double f = 55;                           // focal length in mm
    double sx = 22.3, sy = 14.9;             // sensor size
    double width = 640, height = 480;        // image size

    Mat K = (Mat_<double>(3, 3) << width*f/sx, 0, width/2,   // fx 0 cx
                                 0, height*f/sy, height/2, // 0 fy cx
                                 0, 0, 1);                 // 0  0  1

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


    T = (Mat_<double>(4,4) << R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
                              R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
                              R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0),
                                              0,                 0,                 0,                 1);

    C = C * T;

    R = (Mat_<double>(3,3) << C.at<double>(0,0), C.at<double>(0,1), C.at<double>(0,2),
                              C.at<double>(1,0), C.at<double>(1,1), C.at<double>(1,2),
                              C.at<double>(2,0), C.at<double>(2,1), C.at<double>(2,2));
    
    t = (Mat_<double>(1,3) << C.at<double>(0,3), C.at<double>(1,3), C.at<double>(2,3));
    
    cout << "R: " << std::endl << R <<endl;
    cout << "t: " << std::endl << t <<endl;
    cout << "C: " << std::endl << C <<endl;
    cout << "T: " << std::endl << T <<endl;
    cout << "******************" <<endl;

    /// Construct a cube widget
    viz::WCube cube_widget(Point3f(0.5,0.5,0.0), Point3f(0.0,0.0,-0.5), true, viz::Color::blue());
    cube_widget.setRenderingProperty(viz::LINE_WIDTH, 4.0);

    /// Construct pose
    Affine3d new_cam_pose(R, t);

    /// Display widget (update if already displayed)
    myWindow.showWidget("CPW", cpw, new_cam_pose);
    myWindow.showWidget("CPW_FRUSTUM", cpw_frustum, new_cam_pose);

    /// Event loop for 1 millisecond
    myWindow.spinOnce(1, true);

    Mat res;
    drawMatches(current_frame, inliers1, previous_frame, inliers2, inlier_matches, res);
    imshow("2D Coordinate Frame", res);
    waitKey(1);

    counter++;
  }

}
