#define _USE_MATH_DEFINES
#include <cmath>
#include "tracker.h"
#include "utils.h"
#include <opencv2/aruco.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

namespace {
    // Static variables for tracking state
    bool initialized = false;
    cv::VideoCapture inputVideo;
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::aruco::GridBoard> boards;
    cv::aruco::ArucoDetector detector;

    /**
     * @brief Create marker boards for cube faces
     */
    std::vector<cv::aruco::GridBoard> createBoards(
        float markerSideLength, 
        float markerGapLength, 
        cv::aruco::Dictionary& dictionary, 
        const std::string& boardDirPath
    ) {
        cv::Mat boardImage;
        std::vector<cv::aruco::GridBoard> boards;
        
        for (int i = 0; i < 6; i++) {
            int firstId = i * 4;
            std::vector<int> ids = { firstId, firstId + 1, firstId + 2, firstId + 3 };
            cv::Mat idsMat(ids, true);

            cv::aruco::GridBoard board(
                cv::Size(2, 2), 
                markerSideLength, 
                markerGapLength, 
                dictionary, 
                idsMat
            );
            
            board.generateImage(cv::Size(500, 500), boardImage, 10, 1);
            boards.push_back(board);

            std::string name = boardDirPath + "/board" + std::to_string(i) + ".png";
            cv::imwrite(name, boardImage);
        }
        
        return boards;
    }

    /**
     * @brief Initialize tracking system
     */
    bool initializeTracking(
        double markerSideLength,
        double markerGapLength,
        const std::string& calibrationFilePath,
        const std::string& boardDirPath
    ) {
        if (initialized) {
            return true;
        }

        if (!inputVideo.open(0)) {
            std::cerr << "Could not open camera index 0\n";
            return false;
        }

        if (!utils::readCameraParameters(calibrationFilePath, cameraMatrix, distCoeffs)) {
            std::cerr << "Failed to read camera params from: " << calibrationFilePath << "\n";
            return false;
        }

        cv::aruco::Dictionary dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        boards = createBoards(
            static_cast<float>(markerSideLength),
            static_cast<float>(markerGapLength),
            dict,
            boardDirPath
        );

        cv::aruco::DetectorParameters params;
        detector = cv::aruco::ArucoDetector(dict, params);

        initialized = true;
        return true;
    }

    /**
     * @brief Detect markers in the current camera frame
     */
    bool detectMarkersInFrame(
        cv::Mat& frame,
        std::vector<int>& ids,
        std::vector<std::vector<cv::Point2f>>& corners,
        std::vector<std::vector<cv::Point2f>>& rejected
    ) {
        if (!inputVideo.grab()) {
            std::cerr << "Failed to grab frame.\n";
            return false;
        }
        inputVideo.retrieve(frame);

        detector.detectMarkers(frame, corners, ids, rejected);

        for (size_t i = 0; i < boards.size(); i++) {
            detector.refineDetectedMarkers(frame, boards[i], corners, ids, rejected);
        }

        return true;
    }

    /**
     * @brief Calculate poses for detected boards
     */
    bool calculateBoardPoses(
        const std::vector<int>& ids,
        const std::vector<std::vector<cv::Point2f>>& corners,
        double markerSideLength,
        double markerGapLength,
        std::vector<cv::Vec3d>& foundRvecs,
        std::vector<cv::Vec3d>& foundTvecs
    ) {
        if (ids.empty()) {
            return false;
        }

        for (size_t i = 0; i < boards.size(); i++) {
            cv::Mat objPoints, imgPoints;
            boards[i].matchImagePoints(corners, ids, objPoints, imgPoints);

            if (objPoints.empty() || imgPoints.empty()) {
                continue;
            }

            cv::Vec3d boardR, boardT;
            bool valid = cv::solvePnP(objPoints, imgPoints,
                cameraMatrix, distCoeffs,
                boardR, boardT);

            if (valid) {
                utils::cubeCoordinates(static_cast<int>(i), boardR, boardT,
                    static_cast<float>(markerSideLength),
                    static_cast<float>(markerGapLength));
                foundRvecs.push_back(boardR);
                foundTvecs.push_back(boardT);
            }
        }

        return !foundRvecs.empty();
    }

    /**
     * @brief Draw visualization of detected pose
     */
    void visualizePose(
        cv::Mat& frame,
        const cv::Vec3d& rvec,
        const cv::Vec3d& tvec,
        float markerSideLength,
        bool showVisualization
    ) {
        if (showVisualization) {
            cv::drawFrameAxes(frame, cameraMatrix, distCoeffs,
                rvec, tvec, markerSideLength);
            cv::imshow("out", frame);
            cv::waitKey(1);
        }
    }
}

TrackerPose getCubePose(
    bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string& calibrationFilePath,
    const std::string& boardDirPath
) {
    TrackerPose poseResult = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

    if (!initialized && !initializeTracking(markerSideLength, markerGapLength, 
        calibrationFilePath, boardDirPath)) {
        return poseResult;
    }

    // Detect markers
    cv::Mat frame;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    if (!detectMarkersInFrame(frame, ids, corners, rejected)) {
        return poseResult;
    }

    // Calculate poses
    std::vector<cv::Vec3d> foundRvecs, foundTvecs;
    if (calculateBoardPoses(ids, corners, markerSideLength, markerGapLength, 
        foundRvecs, foundTvecs)) {
        
        cv::Vec3d rvecFinal, tvecFinal;
        utils::averageCube(foundRvecs, foundTvecs, rvecFinal, tvecFinal);

        // Update result
        for (int i = 0; i < 3; i++) {
            poseResult.rotation[i] = rvecFinal[i];
            poseResult.translation[i] = tvecFinal[i];
        }

        visualizePose(frame, rvecFinal, tvecFinal, 
            static_cast<float>(markerSideLength), showVisualization);
    }

    return poseResult;
}

PoseMatrix4x4 getCubePoseMatrix(
    bool showVisualization,
    double markerSideLength,
    double markerGapLength,
    const std::string& calibrationFilePath,
    const std::string& boardDirPath,
    const PoseMatrix4x4& headsetPose,
    ImageData& undistortedImage,
    bool useOSGSpace
) {
    // Initialize tracking if needed
    if (!initializeTracking(markerSideLength, markerGapLength, 
        calibrationFilePath, boardDirPath)) {
        return headsetPose;
    }

    cv::Mat headsetMat = utils::poseMatrixToCvMat(headsetPose);

    // Detect markers
    cv::Mat frame;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners, rejected;
    if (!detectMarkersInFrame(frame, ids, corners, rejected)) {
        return headsetPose;
    }

    // Process undistorted image if needed
    if (showVisualization) {
        cv::Mat cvUndistortedImage;
        cv::undistort(frame, cvUndistortedImage, cameraMatrix, distCoeffs);
        utils::convertMatToImageData(cvUndistortedImage, undistortedImage);
    } else {
        undistortedImage = ImageData();
    }

    // Calculate poses for detected boards
    std::vector<cv::Vec3d> foundRvecs, foundTvecs;
    calculateBoardPoses(ids, corners, markerSideLength, markerGapLength, 
        foundRvecs, foundTvecs);

    // Process final transformation
    cv::Mat finalTransform = cv::Mat::eye(4, 4, CV_64F);
    if (!foundRvecs.empty()) {
        cv::Vec3d rvecFinal, tvecFinal;
        utils::averageCube(foundRvecs, foundTvecs, rvecFinal, tvecFinal);

        // Build cube transformation
        cv::Mat cubePose = utils::buildTransformation(rvecFinal, tvecFinal);
        cv::Mat camPose = utils::poseMatrixToCvMat(headsetPose);

        // Use selected transformation approach
        if (useOSGSpace) {
            finalTransform = utils::transformCubeOSGSpace(camPose, cubePose);
        } else {
            finalTransform = utils::transformCubeOpenCVSpace(camPose, cubePose);
        }

        // Show visualization if requested
        if (showVisualization) {
            visualizePose(frame, rvecFinal, tvecFinal, 
                static_cast<float>(markerSideLength), true);
        }
    } else {
        finalTransform = headsetMat.clone();
    }

    return utils::cvMatToPoseMatrix(finalTransform);
}
