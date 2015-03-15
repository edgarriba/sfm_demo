/*
 * RobustMatcher.h
 *
 *  Created on: Mar 15, 2015
 *  Author: eriba
 */

#include <robust_matcher.h>

RobustMatcher::RobustMatcher() :
  detector_(),
  matcher_()
{
  ransac_thresh_ = 2.5f;
  nn_match_ratio_ = 0.8f;
}


RobustMatcher::RobustMatcher(cv::Ptr<cv::Feature2D> detector, 
    cv::Ptr<cv::DescriptorMatcher> matcher) :
  detector_(detector),
  matcher_(matcher)
{
  ransac_thresh_ = 2.5f;
  nn_match_ratio_ = 0.8f;
}
  
RobustMatcher::~RobustMatcher() {}


// Set the feature detector
void RobustMatcher::setFeatureDetector(const cv::Ptr<cv::Feature2D> &detector)
{  
  detector_ = detector;
}


// Set the matcher
void RobustMatcher::setDescriptorMatcher(const cv::Ptr<cv::DescriptorMatcher> &matcher)
{
  matcher_ = matcher;
}


// Set ratio parameter for the ratio test
void RobustMatcher::setNNMatchRatio(const double nn_match_ratio)
{
  nn_match_ratio_ = nn_match_ratio;
}


// Set RANSAC inlier threshold
void RobustMatcher::setRansacThreshold(const double ransac_thresh)
{
  ransac_thresh_ = ransac_thresh;
}


// Detect the keypoints of an image
void RobustMatcher::detectKeyPoints(const cv::Mat &image, KeyPointVec &kpts)
{
  detector_->detect(image, kpts);
}


// Compute the descriptors of an image given its keypoints
void RobustMatcher::computeDescriptors(const cv::Mat &image, const KeyPointVec &kpts, cv::Mat &desc)
{
  //detector_->compute(image, kpts, desc);
}


// Detect and Compute the keypoints and descriptors of a given image
void RobustMatcher::detectAndCompute(const cv::Mat &image, KeyPointVec &kpts, cv::Mat &desc)
{
  detector_->detectAndCompute(image, cv::noArray(), kpts, desc);
}


// ***********************************************************************


// Match features and get essential mat
bool RobustMatcher::robustMatchEssentialMat(const cv::Mat &frame1, const cv::Mat &frame2,
  KeyPointVec &kpts1_inliers, KeyPointVec &kpts2_inliers,
  DMatchVec &inliers_matches, cv::Mat &essentialMat)
{
  cv::Mat desc1, desc2, inliers_mask;
  KeyPointVec kpts1, kpts1_good, kpts2, kpts2_good;
  DMatchVec good_matches;
  DMatchVec2 nn_matches;

  // compute keypoints and decriptor and match
  nnMatch(frame1, kpts1, desc1, frame2, kpts2, desc2, nn_matches);

  // perform ratio test
  ratioTest(nn_matches, good_matches, kpts1, kpts1_good, kpts2, kpts2_good);

  // for essential mat we need
  // at least 4 points
  if(kpts2_good.size() >= 4)
  {
    computeEssentialMat(kpts1_good, kpts2_good, essentialMat, inliers_mask);
  }
  else
  {
    return false;
  }

  // extract inliers
  extractInliers(inliers_mask, kpts1_good, kpts2_good, kpts1_inliers, kpts2_inliers, inliers_matches);

  return true;
}


// ***********************************************************************


// Does NN Match
void RobustMatcher::nnMatch(const cv::Mat &frame1, KeyPointVec &kpts1, cv::Mat &desc1,
                            const cv::Mat &frame2, KeyPointVec &kpts2, cv::Mat &desc2,
                            DMatchVec2 &nn_matches)
{

  // Detect ans compute keypoints and descriptors
  detectAndCompute(frame1, kpts1, desc1);
  detectAndCompute(frame2, kpts2, desc2);


  // Match the two image descriptors
  // return 2 nearest neighbours
  matcher_->knnMatch(desc1, desc2, nn_matches, 2);

}


// ***********************************************************************



// Remove matches for which NN ratio is > than threshold
// return the number of removed points
int RobustMatcher::ratioTest(const DMatchVec2 &matches, DMatchVec &good_matches,
  const KeyPointVec &kpts1, KeyPointVec &kpts1_out,
  const KeyPointVec &kpts2, KeyPointVec &kpts2_out)
{
  for(size_t i = 0; i < matches.size(); i++)
  {
    cv::DMatch first = matches[i][0];
    float dist1 = matches[i][0].distance;
    float dist2 = matches[i][1].distance;

    if(dist1 < nn_match_ratio_ * dist2) {
      kpts1_out.push_back(kpts1[first.queryIdx]);
      kpts2_out.push_back(kpts2[first.trainIdx]);
      good_matches.push_back(first);
    }
  }
  return static_cast<int>(good_matches.size());
}


// ***********************************************************************


// Compute the essential matrix given a set of 
// paired keypoints list
void RobustMatcher::computeEssentialMat(const KeyPointVec &kpts1,
  const KeyPointVec &kpts2, cv::Mat &essentialMat, cv::Mat &inliers_mask)
{
  std::vector<cv::Point2f> kpts1_pts, kpts2_pts;

  // put keypoints inside vector
  for (int i = 0; i < kpts1.size(); ++i)
  {
    kpts1_pts.push_back(kpts1[i].pt);
    kpts2_pts.push_back(kpts2[i].pt);
  }

  // default params
  double focal = 1.0;
  cv::Point2d pp = cv::Point2d(0,0);
  int method = cv::RANSAC;
  double prob = 0.999;
  double threshold = 1.0;
  
  // compute opencv essential mat
  essentialMat = 
    findEssentialMat(kpts1_pts, kpts2_pts, focal, pp, method, prob, ransac_thresh_, inliers_mask);

}


// ***********************************************************************


// Extract the inliers keypoints given a mask
void RobustMatcher::extractInliers(const cv::Mat &inliers_mask,
  const KeyPointVec &kpts1, const KeyPointVec &kpts2,
  KeyPointVec &kpts1_inliers, KeyPointVec &kpts2_inliers,
  DMatchVec &inliers_matches)
{
  for(unsigned i = 0; i < kpts1.size(); i++)
  {
    if(inliers_mask.at<uchar>(i)) {
        int new_i = static_cast<int>(kpts1_inliers.size());
        kpts1_inliers.push_back(kpts1[i]);
        kpts2_inliers.push_back(kpts2[i]);
        inliers_matches.push_back(cv::DMatch(new_i, new_i, 0));
    }
  }

}