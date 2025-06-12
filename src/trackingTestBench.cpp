#include "utils.h"
#include "tracker.h"
#include <iostream>
#include <iomanip>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

void printMatrix(const PoseMatrix4x4& matrix, const std::string& name) {
    std::cout << "\n" << name << ":\n";
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(4) 
                     << matrix.m[i * 4 + j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

void testTransformationApproaches() {
    std::cout << "Testing Cube Pose Transformation Approaches\n";
    std::cout << "=========================================\n";

    // Create a test camera pose in OSG space (at position (0,2,2) looking at origin)
    PoseMatrix4x4 osgCameraPose;
    for (int i = 0; i < 16; i++) {
        osgCameraPose.m[i] = (i % 5 == 0) ? 1.0 : 0.0;
    }
    osgCameraPose.m[3] = 0.0;   // X
    osgCameraPose.m[7] = 2.0;   // Y (forward in OSG)
    osgCameraPose.m[11] = 2.0;  // Z (up in OSG)

    // Create a test cube pose in OpenCV space (1 meter in front of camera)
    cv::Vec3d cubeRvec(0, 0, 0);  // No rotation
    cv::Vec3d cubeTvec(0, 0, 1);  // 1 meter in Z (forward in OpenCV)
    cv::Mat cubePoseCV = utils::buildTransformation(cubeRvec, cubeTvec);
    cv::Mat cameraPoseCV = utils::poseMatrixToCvMat(osgCameraPose);

    // Test OSG Space Approach
    std::cout << "\nTesting OSG Space Approach:\n";
    std::cout << "===========================\n";
    cv::Mat resultOSG = utils::transformCubeOSGSpace(cameraPoseCV, cubePoseCV);
    printMatrix(utils::cvMatToPoseMatrix(resultOSG), "Result (OSG Space)");

    // Test OpenCV Space Approach
    std::cout << "\nTesting OpenCV Space Approach:\n";
    std::cout << "==============================\n";
    cv::Mat resultCV = utils::transformCubeOpenCVSpace(cameraPoseCV, cubePoseCV);
    printMatrix(utils::cvMatToPoseMatrix(resultCV), "Result (OpenCV Space)");

    // Print difference between approaches
    cv::Mat diff = resultOSG - resultCV;
    double maxDiff = cv::norm(diff, cv::NORM_INF);
    std::cout << "\nMaximum difference between approaches: " << maxDiff << std::endl;

    std::cout << "\nExpected behavior:\n";
    std::cout << "- Both approaches should place the cube in world space\n";
    std::cout << "- The cube should maintain its relative position to the camera\n";
    std::cout << "- The differences between approaches should be minimal\n";
    std::cout << "- Check if the cube appears correctly oriented in your OSG viewer\n";
}

void testRealTimeComparison() {
    std::cout << "\nRunning Real-time Test\n";
    std::cout << "=====================\n";
    
    const std::string calibFile = "data/camera_calibration/camera_calibration.txt";
    const std::string boardDir = "data/markers/runtime";
    const double markerSize = 0.04;  // 4cm
    const double gapSize = 0.01;     // 1cm
    
    ImageData imgData;
    bool showViz = true;
    bool useOSG = true;  // Can be changed manually here to switch between approaches

    std::cout << "Press Q to quit\n\n";

    char key = 0;
    while (key != 'q' && key != 'Q') {
        // Create a test camera pose
        PoseMatrix4x4 testCameraPose;
        for (int i = 0; i < 16; i++) {
            testCameraPose.m[i] = (i % 5 == 0) ? 1.0 : 0.0;
        }

        // Get cube pose using the selected approach
        PoseMatrix4x4 result = getCubePoseMatrix(
            showViz,
            markerSize,
            gapSize,
            calibFile,
            boardDir,
            testCameraPose,
            imgData,
            useOSG
        );

        printMatrix(result, "Resulting Pose");
        key = cv::waitKey(10);
    }
}

int main() {
    std::cout << "Cube Pose Transformation\n";
    std::cout << "=======================\n\n";
    
    testRealTimeComparison();
    return 0;
}
