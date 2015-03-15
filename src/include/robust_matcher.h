/*
 * RobustMatcher.h
 *
 *  Created on: Mar 15, 2015
 *  Author: eriba
 */

#ifndef ROBUSTMATCHER_H_
#define ROBUSTMATCHER_H_

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>// move to visualizer
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d.hpp>

typedef std::vector<cv::KeyPoint> KeyPointVec;
typedef std::vector<cv::DMatch> DMatchVec;
typedef std::vector<std::vector<cv::DMatch> > DMatchVec2;

class RobustMatcher {
public:
  RobustMatcher();
  RobustMatcher(cv::Ptr<cv::Feature2D> detector, cv::Ptr<cv::DescriptorMatcher> matcher);
  
  ~RobustMatcher();

  // Set the feature detector
  void setFeatureDetector(const cv::Ptr<cv::Feature2D> &detector);

  // Set the matcher
  void setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher> &matcher);

  // Set ratio parameter for the ratio test
  void setNNMatchRatio(const double nn_match_ratio);

  // Set RANSAC inlier threshold
  void setRansacThreshold(const double ransac_thresh);

  // Detect the keypoints of an image
  void detectKeyPoints(const cv::Mat &image, KeyPointVec &kpts);

  // Compute the descriptors of an image given its keypoints
  void computeDescriptors(const cv::Mat &image, const KeyPointVec &kpts, cv::Mat &desc);

  // Detect and Compute the keypoints and descriptors of a given image
  void detectAndCompute(const cv::Mat &image, KeyPointVec &kpts, cv::Mat &desc);


  // Match features and get essential mat
  bool robustMatchEssentialMat(const cv::Mat &frame1, const cv::Mat &frame2,
    KeyPointVec &kpts1_inliers, KeyPointVec &kpts2_inliers,
    DMatchVec &inliers_matches, cv::Mat &essentialMat);


  // Does NN Match
  void nnMatch(const cv::Mat &frame1, KeyPointVec &kpts1, cv::Mat &desc1,
               const cv::Mat &frame2, KeyPointVec &kpts2, cv::Mat &desc2,
               DMatchVec2 &nn_matches);


protected:

  // Remove matches for which NN ratio is > than threshold
  // return the number of removed points
  int ratioTest(const DMatchVec2 &matches, DMatchVec &good_matches,
    const KeyPointVec &kpts1, KeyPointVec &kpts1_out,
    const KeyPointVec &kpts2, KeyPointVec &kpts2_out);

  // Compute the essential matrix given a set of 
  // paired keypoints list
  void computeEssentialMat(const KeyPointVec &kpts1,
    const KeyPointVec &kpts2, cv::Mat &essentialMat, cv::Mat &inliers_mask);


  // Extract the inliers keypoints given a mask
  void extractInliers(const cv::Mat &inliers_mask,
    const KeyPointVec &kpts1, const KeyPointVec &kpts2,
    KeyPointVec &kpts1_inliers, KeyPointVec &kpts2_inliers,
    DMatchVec &inliers_matches);

private:
  // pointer to the feature point detector object
  cv::Ptr<cv::Feature2D> detector_;
  // pointer to the matcher object
  cv::Ptr<cv::DescriptorMatcher> matcher_;
  // RANSAC inlier threshold
  double ransac_thresh_;
  // Nearest-neighbour matching ratio
  double nn_match_ratio_;
};

#endif /* ROBUSTMATCHER_H_ */
