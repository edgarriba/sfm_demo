#include <iostream>

using namespace std;

/**  GLOBAL VARIABLES  **/

string tutorial_path = "../";
string img1_path = tutorial_path + "data/box_pose1.JPG";  // image 1 path
string img2_path = tutorial_path + "data/box_pose2.JPG";  // image 2 path

string mesh_path = tutorial_path + "data/horse.ply"; // box mesh path

// Intrinsic camera parameters: UVC WEBCAM
double f = 55;                           // focal length in mm
double sx = 22.3, sy = 14.9;             // sensor size
double width = 640, height = 480;        // image size

double params_WEBCAM[] = { width*f/sx,   // fx
                           height*f/sy,  // fy
                           width/2,      // cx
                           height/2};    // cy


const double akaze_thresh = 3e-4;   // AKAZE detection threshold set to locate about 1000 keypoints
const double ransac_thresh = 2.5f;  // RANSAC inlier threshold
const double nn_match_ratio = 0.8f; // Nearest-neighbour matching ratio


