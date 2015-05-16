#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>

#include <chrono>
#include <iostream>

int main(int argc, char **argv) {
    if (argc == 3) {
        cv::Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
        cv::Mat img_2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
        if (!img_1.data || !img_2.data)
            return -1;
        cv::SiftFeatureDetector detector;
        std::vector<cv::KeyPoint> key_pt_1, key_pt_2;
        detector.detect(img_1, key_pt_1);
        detector.detect(img_2, key_pt_2);
        cv::SiftDescriptorExtractor extractor;
        cv::Mat descriptors_1, descriptors_2;
        extractor.compute(img_1, key_pt_1, descriptors_1);
        extractor.compute(img_2, key_pt_2, descriptors_2);
        std::chrono::high_resolution_clock::time_point t1 =
                std::chrono::high_resolution_clock::now();
        std::vector<cv::DMatch> matches;
        std::vector<cv::DMatch> good_matches;
        cv::FlannBasedMatcher matcher;
        matcher.match(descriptors_1, descriptors_2, matches);
        double min_dst = 1000.0, max_dst = -1000.0;
        for (size_t i = 0; i < matches.size(); ++i) {
            double dst = matches[i].distance;
            if (dst < min_dst)
                min_dst = dst;
            if (dst > max_dst)
                max_dst = dst;
        }
        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].distance < std::max(3 * min_dst, 0.02))
                good_matches.push_back(matches[i]);
        }

        std::chrono::high_resolution_clock::time_point t2 =
                std::chrono::high_resolution_clock::now();
        std::cout << "Matches amount = " << matches.size() << std::endl;
        std::cout << "Time = " << std::chrono::duration_cast<
                                        std::chrono::milliseconds>(t2 - t1).count() << std::endl;
        cv::Mat out_img;
        cv::drawMatches(img_1, key_pt_1, img_2, key_pt_2, good_matches, out_img,
                        cv::Scalar::all(-1), cv::Scalar::all(-1),
                        std::vector<char>(),  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imwrite("matches.jpg", out_img);
        return 0;
    }
}
