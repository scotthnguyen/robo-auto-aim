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

  // Decides whether the plate in `box` is lit red or blue by sampling the strip
  // bands at its left and right edges. `img` must be rgb8. Returns RED, BLUE,
  // or -1 when nothing in the box is lit brightly enough to call.
  static int sampleLightColor(const cv::Mat & img, const cv::Rect & box);

private:
  void preprocess(
    const cv::Mat & img, cv::Mat & blob, float & scale, cv::Point & offset) const;

  std::vector<Armor> postprocess(
    const cv::Mat & output, const cv::Mat & img, float scale, const cv::Point & offset) const;

  static Armor boxToArmor(const cv::Rect & box, int class_id, int color);

  cv::dnn::Net net_;
  float conf_threshold_;
  float nms_threshold_;

  // YOLOv8 input dimensions
  static constexpr int INPUT_W = 640;
  static constexpr int INPUT_H = 640;
  // 0 = small_armor, 1 = large_armor
  static constexpr int NUM_CLASSES = 2;
  // Minimum R or B value for a pixel to count as "lit" when sampling colour
  static constexpr int LIGHT_MIN_INTENSITY = 120;
};

}  // namespace sn_auto_aim

#endif  // ARMOR_DETECTOR__YOLO_DETECTOR_HPP_
