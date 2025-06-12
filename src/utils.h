#ifndef UTILS_H
#define UTILS_H

#include <opencv2/core.hpp>
#include "tracker.h"

namespace utils {

// Matrix conversion utilities
cv::Mat poseMatrixToCvMat(const PoseMatrix4x4& pose);
PoseMatrix4x4 cvMatToPoseMatrix(const cv::Mat& mat);
cv::Mat buildTransformation(const cv::Vec3d& rvec, const cv::Vec3d& tvec);

// Coordinate transformation utilities
cv::Mat getCoordinateConversionMatrix();
cv::Mat transformCubeOpenCVSpace(const cv::Mat& cameraPose, const cv::Mat& cubePose);
cv::Mat transformCubeOSGSpace(const cv::Mat& cameraPose, const cv::Mat& cubePose);

// OSG conversion utilities
PoseMatrix4x4 convertOSGCameraToCubePose(const PoseMatrix4x4& osgCameraPose);

// Image conversion utilities
void convertMatToImageData(const cv::Mat& mat, ImageData& imageData);

// Rotation utilities
cv::Vec3d moveAxis(cv::Vec3d& tvec, cv::Vec3d rvec, double distance, int axis);
cv::Vec3d rotateXAxis(cv::Vec3d rotation, double angleRad);
cv::Vec3d rotateYAxis(cv::Vec3d rotation, double angleRad);
cv::Vec3d rotateZAxis(cv::Vec3d rotation, double angleRad);

// Camera utilities
bool readCameraParameters(const std::string& filename, cv::Mat& camMatrix, cv::Mat& distCoeffs);

// Pose calculation utilities
void averageCube(std::vector<cv::Vec3d>& rvecs, std::vector<cv::Vec3d>& tvecs, 
                 cv::Vec3d& outputRvec, cv::Vec3d& outputTvec);
void cubeCoordinates(int id, cv::Vec3d& rvecs, cv::Vec3d& tvecs, 
                    float markerSideLength, float markerGapLength);

} // namespace utils

#endif // UTILS_H
