// Scott Nguyen — MIT License

#ifndef ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
#define ARMOR_DETECTOR__YOLO_DETECTOR_HPP_

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

#include "armor_detector/armor.hpp"

namespace sn_auto_aim
{

class YoloDetector
{
public:
  YoloDetector(
    const std::string & model_path, float conf_threshold = 0.5f, float nms_threshold = 0.45f);

  std::vector<Armor> detect(const cv::Mat & img);

private:
  void preprocess(
    const cv::Mat & img, cv::Mat & blob, float & scale, cv::Point & offset) const;

  std::vector<Armor> postprocess(
    const cv::Mat & output, float scale, const cv::Point & offset) const;

  static Armor boxToArmor(const cv::Rect & box, int class_id);

  cv::dnn::Net net_;
  float conf_threshold_;
  float nms_threshold_;

  // YOLOv8 input dimensions
  static constexpr int INPUT_W = 640;
  static constexpr int INPUT_H = 640;
  // 0 = small_armor, 1 = large_armor
  static constexpr int NUM_CLASSES = 2;
};

}  // namespace sn_auto_aim

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
