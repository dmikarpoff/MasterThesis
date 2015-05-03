#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <chrono>
#include <iostream>

int main(int argc, char **argv) {
    if (argc == 2) {
        cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        if (!img_1.data)
            return -1;
        std::chrono::high_resolution_clock::time_point t1 =
                std::chrono::high_resolution_clock::now();
        cv::SiftFeatureDetector detector;
        std::vector<cv::KeyPoint> key_pt;
        detector.detect(img_1, key_pt);
        cv::SiftDescriptorExtractor extractor;
        cv::Mat descriptors;
        extractor.compute(img_1, key_pt, descriptors);
        std::chrono::high_resolution_clock::time_point t2 =
                std::chrono::high_resolution_clock::now();
        std::cout << "Key point size = " << key_pt.size() << std::endl;
        std::cout << "Time = " << std::chrono::duration_cast<
                                        std::chrono::milliseconds>(t2 - t1).count() << std::endl;
        cv::Mat out_img;
        cv::drawKeypoints(img_1, key_pt, out_img, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imwrite("sift.jpg", out_img);
        cv::waitKey(5);
        return 0;
    }
}
