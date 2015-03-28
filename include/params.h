#include <iostream>
#include <opencv2/core.hpp>

/**  GLOBAL VARIABLES  **/

std::string tutorial_path = "../";
std::string img1_path = tutorial_path + "data/box_pose1.JPG";  // image 1 path
std::string img2_path = tutorial_path + "data/box_pose2.JPG";  // image 2 path
std::string video1_path = tutorial_path + "data/box.mp4";  // image 2 path
std::string video2_path = tutorial_path + "data/box2.mp4";  // image 2 path

std::string mesh_path = tutorial_path + "data/horse.ply"; // box mesh path

// Intrinsic camera parameters: UVC WEBCAM
double f = 55;                           // focal length in mm
double sx = 22.3, sy = 14.9;             // sensor size
double width = 640, height = 480;        // image size

// Camera Matrix
cv::Mat K = (cv::Mat_<double>(3, 3) << width*f/sx, 0, width/2,  // fx 0 cx
                                     0, height*f/sy, height/2,  // 0 fy cx
                                                      0, 0, 1); // 0  0  1

const double akaze_thresh = 3e-4;   // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f;  // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio


