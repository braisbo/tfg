#ifndef TRACKER_H
#define TRACKER_H

#include <string>
#include <vector>

/**
 * @brief Represents the pose (rotation and translation) of the tracked object
 */
struct TrackerPose {
    double rotation[3];     ///< Rotation vector (Rodrigues format)
    double translation[3];  ///< Translation vector (X, Y, Z)
};

/**
 * @brief Represents a 4x4 transformation matrix in column-major order
 */
struct PoseMatrix4x4 {
    double m[16];  ///< Matrix elements in column-major order
};

/**
 * @brief Container for image data with basic properties
 */
struct ImageData {
    std::vector<unsigned char> data;  ///< Raw image data
    int width;                        ///< Image width in pixels
    int height;                       ///< Image height in pixels
    int channels;                     ///< Number of color channels (3 for RGB, 4 for RGBA)
    
    /**
     * @brief Check if the image contains any data
     * @return true if the image is empty, false otherwise
     */
    bool isEmpty() const { return data.empty(); }
};

/**
 * @brief Get the pose of a cube in camera space using marker detection
 * 
 * @param showVisualization Whether to show visual debugging output
 * @param markerSideLength Length of each marker's side in meters
 * @param markerGapLength Gap between markers in meters
 * @param calibrationFilePath Path to the camera calibration file
 * @param boardDirPath Directory path where marker board images will be saved
 * @return TrackerPose containing rotation and translation of the cube
 */
TrackerPose getCubePose(
    bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string& calibrationFilePath,
    const std::string& boardDirPath
);

/**
 * @brief Get the cube's pose as a 4x4 matrix in world space
 * 
 * @param showVisualization Whether to show visual debugging output
 * @param markerSideLength Length of each marker's side in meters
 * @param markerGapLength Gap between markers in meters
 * @param calibrationFilePath Path to the camera calibration file
 * @param boardDirPath Directory path where marker board images will be saved
 * @param headsetPose Current pose of the VR headset in world space
 * @param undistortedImage Output parameter for the undistorted camera image (if visualization is enabled)
 * @return PoseMatrix4x4 containing the cube's transformation in world space
 */
PoseMatrix4x4 getCubePoseMatrix(
    bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string& calibrationFilePath,
    const std::string& boardDirPath,
    const PoseMatrix4x4& headsetPose,
    ImageData& undistortedImage,
    bool useOSGSpace = true
);

#endif // TRACKER_H
