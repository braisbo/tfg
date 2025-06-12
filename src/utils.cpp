#define _USE_MATH_DEFINES
#include <cmath>
#include "utils.h"
#include <opencv2/calib3d.hpp>
#include <cstring>

namespace utils {

cv::Mat getCoordinateConversionMatrix() {
    cv::Mat coordConv = cv::Mat::zeros(4, 4, CV_64F);
    coordConv.at<double>(0,0) = 1.0;  // X stays same
    coordConv.at<double>(1,2) = 1.0;  // Y -> Z
    coordConv.at<double>(2,1) = -1.0; // Z -> -Y
    coordConv.at<double>(3,3) = 1.0;
    return coordConv;
}

cv::Mat transformCubeOpenCVSpace(const cv::Mat& cameraPose, const cv::Mat& cubePose) {
    cv::Mat coordConv = getCoordinateConversionMatrix();
    
    // Convert camera to OpenCV space
    cv::Mat cameraInOpenCV = coordConv * cameraPose * coordConv.inv();
    
    // Multiply with cube pose
    cv::Mat cubeWorldPose = cameraInOpenCV * cubePose;
    
    // Convert back to OSG space
    return coordConv.inv() * cubeWorldPose * coordConv;
}

cv::Mat transformCubeOSGSpace(const cv::Mat& cameraPose, const cv::Mat& cubePose) {
    cv::Mat coordConv = getCoordinateConversionMatrix();
    
    // Convert cube to OSG space and combine with camera
    return cameraPose * (coordConv.inv() * cubePose * coordConv);
}


cv::Mat poseMatrixToCvMat(const PoseMatrix4x4& pose) {
    cv::Mat mat(4, 4, CV_64F);
    for (int i = 0; i < 16; i++) {
        mat.at<double>(i / 4, i % 4) = pose.m[i];
    }
    return mat;
}

PoseMatrix4x4 cvMatToPoseMatrix(const cv::Mat& mat) {
    PoseMatrix4x4 out;
    for (int row = 0; row < 4; row++) {
        for (int col = 0; col < 4; col++) {
            out.m[row * 4 + col] = mat.at<double>(row, col);
        }
    }
    return out;
}

PoseMatrix4x4 convertOSGCameraToCubePose(const PoseMatrix4x4& osgCameraPose) {
    // Convert OSG camera pose (which is in OSG world space) to cube pose
    
    // First convert to cv::Mat for easier manipulation
    cv::Mat cameraPose = poseMatrixToCvMat(osgCameraPose);
    
    // Create coordinate system conversion matrix (OSG to OpenCV)
    // OSG: Z-up, Y-forward to OpenCV: Z-forward, Y-down
    cv::Mat coordConv = cv::Mat::zeros(4, 4, CV_64F);
    coordConv.at<double>(0,0) = 1.0;  // X stays the same
    coordConv.at<double>(1,2) = 1.0;  // Y -> Z
    coordConv.at<double>(2,1) = -1.0; // Z -> -Y
    coordConv.at<double>(3,3) = 1.0;
    
    // Place cube 1 meter in front of the camera
    cv::Mat translation = cv::Mat::eye(4, 4, CV_64F);
    translation.at<double>(2,3) = 1.0; // Z = 1 meter
    
    // Compute final cube pose:
    // 1. Convert to OpenCV coordinate system
    // 2. Apply translation to place cube in front of camera
    // 3. Convert back to OSG coordinate system
    cv::Mat cubePose = coordConv.inv() * translation * coordConv * cameraPose;
    
    // Convert back to PoseMatrix4x4
    return cvMatToPoseMatrix(cubePose);
}

cv::Mat buildTransformation(const cv::Vec3d& rvec, const cv::Vec3d& tvec) {
    cv::Mat rot3x3;
    cv::Rodrigues(rvec, rot3x3);

    cv::Mat out = cv::Mat::eye(4, 4, CV_64F);
    rot3x3.copyTo(out(cv::Rect(0, 0, 3, 3)));
    out.at<double>(0, 3) = tvec[0];
    out.at<double>(1, 3) = tvec[1];
    out.at<double>(2, 3) = tvec[2];
    return out;
}

void convertMatToImageData(const cv::Mat& mat, ImageData& imageData) {
    if (mat.empty()) {
        imageData.data.clear();
        imageData.width = 0;
        imageData.height = 0;
        imageData.channels = 0;
        return;
    }

    imageData.width = mat.cols;
    imageData.height = mat.rows;
    imageData.channels = mat.channels();

    size_t totalSize = mat.total() * mat.elemSize();
    imageData.data.resize(totalSize);

    if (mat.isContinuous()) {
        std::memcpy(imageData.data.data(), mat.data, totalSize);
    } else {
        size_t rowSize = mat.cols * mat.elemSize();
        for (int i = 0; i < mat.rows; i++) {
            std::memcpy(imageData.data.data() + i * rowSize,
                       mat.ptr(i), rowSize);
        }
    }
}

cv::Vec3d moveAxis(cv::Vec3d& tvec, cv::Vec3d rvec, double distance, int axis) {
    cv::Mat rotationMatrix, rotationMatrixTransposed;
    cv::Rodrigues(rvec, rotationMatrix);
    rotationMatrixTransposed = rotationMatrix.t();
    double* rz = rotationMatrixTransposed.ptr<double>(axis); // x=0, y=1, z=2
    tvec[0] -= rz[0] * distance;
    tvec[1] -= rz[1] * distance;
    tvec[2] -= rz[2] * distance;
    return tvec;
}

cv::Vec3d rotateXAxis(cv::Vec3d rotation, double angleRad) {
    cv::Mat R(3, 3, CV_64F);
    cv::Rodrigues(rotation, R);
    cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
    RX.at<double>(1, 1) = cos(angleRad);
    RX.at<double>(1, 2) = -sin(angleRad);
    RX.at<double>(2, 1) = sin(angleRad);
    RX.at<double>(2, 2) = cos(angleRad);
    R = R * RX;
    cv::Vec3d output;
    cv::Rodrigues(R, output);
    return output;
}

cv::Vec3d rotateYAxis(cv::Vec3d rotation, double angleRad) {
    cv::Mat R(3, 3, CV_64F);
    cv::Rodrigues(rotation, R);
    cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
    RX.at<double>(0, 0) = cos(angleRad);
    RX.at<double>(0, 2) = sin(angleRad);
    RX.at<double>(2, 0) = -sin(angleRad);
    RX.at<double>(2, 2) = cos(angleRad);
    R = R * RX;
    cv::Vec3d output;
    cv::Rodrigues(R, output);
    return output;
}

cv::Vec3d rotateZAxis(cv::Vec3d rotation, double angleRad) {
    cv::Mat R(3, 3, CV_64F);
    cv::Rodrigues(rotation, R);
    cv::Mat RX = cv::Mat::eye(3, 3, CV_64F);
    RX.at<double>(0, 0) = cos(angleRad);
    RX.at<double>(0, 1) = -sin(angleRad);
    RX.at<double>(1, 0) = sin(angleRad);
    RX.at<double>(1, 1) = cos(angleRad);
    R = R * RX;
    cv::Vec3d output;
    cv::Rodrigues(R, output);
    return output;
}

bool readCameraParameters(const std::string& filename, cv::Mat& camMatrix, cv::Mat& distCoeffs) {
    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened())
        return false;
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    return true;
}

void averageCube(std::vector<cv::Vec3d>& rvecs, std::vector<cv::Vec3d>& tvecs, 
                 cv::Vec3d& outputRvec, cv::Vec3d& outputTvec) {
    cv::Vec3d rvecAverage(0, 0, 0);
    cv::Vec3d tvecAverage(0, 0, 0);
    
    for (const auto& rvec : rvecs) {
        for (int j = 0; j < 3; j++) {
            rvecAverage[j] += rvec[j];
        }
    }
    
    for (const auto& tvec : tvecs) {
        for (int j = 0; j < 3; j++) {
            tvecAverage[j] += tvec[j];
        }
    }
    
    for (int x = 0; x < 3; x++) {
        rvecAverage[x] /= rvecs.size();
        tvecAverage[x] /= tvecs.size();
    }
    
    outputRvec = rvecAverage;
    outputTvec = tvecAverage;
    rvecs.clear();
    tvecs.clear();
}

void cubeCoordinates(int id, cv::Vec3d& rvecs, cv::Vec3d& tvecs, 
                    float markerSideLength, float markerGapLength) {
    float boardSideLength = (2 * markerSideLength + markerGapLength) / 2.0f;
    
    // Move to cube center
    tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 0);
    tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 1);
    tvecs = moveAxis(tvecs, rvecs, -boardSideLength, 2);
    
    // Apply face-specific rotations
    switch (id) {
        case 1:
            rvecs = rotateXAxis(rvecs, -M_PI / 2);
            break;
        case 2:
            rvecs = rotateXAxis(rvecs, M_PI);
            break;
        case 3:
            rvecs = rotateYAxis(rvecs, M_PI / 2);
            rvecs = rotateZAxis(rvecs, M_PI);
            break;
        case 4:
            rvecs = rotateXAxis(rvecs, M_PI);
            rvecs = rotateYAxis(rvecs, -M_PI / 2);
            break;
        case 5:
            rvecs = rotateXAxis(rvecs, M_PI / 2);
            break;
        default:
            break;
    }
}

} // namespace utils
