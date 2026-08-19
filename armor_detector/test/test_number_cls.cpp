

#include <gtest/gtest.h>

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/core/mat.hpp>

// STL
#include <algorithm>
#include <cfloat>
#include <chrono>
#include <iostream>
#include <map>
#include <vector>

#include "armor_detector/number_classifier.hpp"

using hrc = std::chrono::high_resolution_clock;

TEST(test_nc, benchmark)
{
  auto pkg_path = ament_index_cpp::get_package_share_directory("armor_detector");
  auto model_path = pkg_path + "/model/mlp.onnx";
  auto label_path = pkg_path + "/model/label.txt";
  sn_auto_aim::NumberClassifier nc(model_path, label_path, 0.5);

  // Matches the ROI the classifier is fed at runtime: 20 wide x 28 tall.
  auto test_mat = cv::Mat::zeros(28, 20, CV_8UC1);

  int loop_num = 200;
  int warm_up = 30;

  double time_min = DBL_MAX;
  double time_max = -DBL_MAX;
  double time_avg = 0;

  for (int i = 0; i < warm_up + loop_num; i++) {
    // classify() erases armors below the confidence threshold, so the input has
    // to be rebuilt each pass or every iteration after the first is a no-op.
    auto dummy_armors = std::vector<sn_auto_aim::Armor>(1);
    dummy_armors[0].number_img = test_mat;
    dummy_armors[0].type = sn_auto_aim::ArmorType::SMALL;

    auto start = hrc::now();
    nc.classify(dummy_armors);
    auto end = hrc::now();
    double time = std::chrono::duration<double, std::milli>(end - start).count();
    if (i >= warm_up) {
      time_min = std::min(time_min, time);
      time_max = std::max(time_max, time);
      time_avg += time;
    }
  }
  time_avg /= loop_num;

  // A no-op loop would report ~0ms; assert the forward pass actually ran.
  EXPECT_GT(time_avg, 0.0);

  std::cout << "time_min: " << time_min << "ms" << std::endl;
  std::cout << "time_max: " << time_max << "ms" << std::endl;
  std::cout << "time_avg: " << time_avg << "ms" << std::endl;
}
