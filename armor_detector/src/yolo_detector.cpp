// Scott Nguyen — MIT License

#include "armor_detector/yolo_detector.hpp"

#include <algorithm>
#include <cmath>

namespace sn_auto_aim
{

YoloDetector::YoloDetector(
  const std::string & model_path, float conf_threshold, float nms_threshold)
: conf_threshold_(conf_threshold), nms_threshold_(nms_threshold)
{
  net_ = cv::dnn::readNetFromONNX(model_path);
}

std::vector<Armor> YoloDetector::detect(const cv::Mat & img)
{
  cv::Mat blob;
  float scale;
  cv::Point offset;
  preprocess(img, blob, scale, offset);

  net_.setInput(blob);
  std::vector<cv::Mat> outputs;
  net_.forward(outputs, net_.getUnconnectedOutLayersNames());

  return postprocess(outputs[0], scale, offset);
}

void YoloDetector::preprocess(
  const cv::Mat & img, cv::Mat & blob, float & scale, cv::Point & offset) const
{
  // Letterbox: scale to fit 640x640 while keeping aspect ratio
  float w_scale = static_cast<float>(INPUT_W) / img.cols;
  float h_scale = static_cast<float>(INPUT_H) / img.rows;
  scale = std::min(w_scale, h_scale);

  int new_w = static_cast<int>(img.cols * scale);
  int new_h = static_cast<int>(img.rows * scale);
  offset.x = (INPUT_W - new_w) / 2;
  offset.y = (INPUT_H - new_h) / 2;

  cv::Mat resized;
  cv::resize(img, resized, cv::Size(new_w, new_h));

  cv::Mat padded(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(114, 114, 114));
  resized.copyTo(padded(cv::Rect(offset.x, offset.y, new_w, new_h)));

  cv::dnn::blobFromImage(
    padded, blob, 1.0 / 255.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), true, false, CV_32F);
}

std::vector<Armor> YoloDetector::postprocess(
  const cv::Mat & output, float scale, const cv::Point & offset) const
{
  // YOLOv8 ONNX output: [1, 4+NUM_CLASSES, 8400]
  // Reshape to [4+NUM_CLASSES, 8400] then transpose to [8400, 4+NUM_CLASSES]
  cv::Mat out = output.reshape(1, output.size[1]);
  cv::transpose(out, out);

  std::vector<cv::Rect> boxes;
  std::vector<float> scores;
  std::vector<int> class_ids;

  for (int i = 0; i < out.rows; i++) {
    const float * row = out.ptr<float>(i);

    // Find highest class score
    float max_score = 0.0f;
    int class_id = 0;
    for (int c = 0; c < NUM_CLASSES; c++) {
      if (row[4 + c] > max_score) {
        max_score = row[4 + c];
        class_id = c;
      }
    }

    if (max_score < conf_threshold_) continue;

    // cx, cy, w, h are in 640x640 letterbox space — unscale back to original image
    float cx = (row[0] - offset.x) / scale;
    float cy = (row[1] - offset.y) / scale;
    float w = row[2] / scale;
    float h = row[3] / scale;

    boxes.emplace_back(
      static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
      static_cast<int>(w), static_cast<int>(h));
    scores.push_back(max_score);
    class_ids.push_back(class_id);
  }

  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, scores, conf_threshold_, nms_threshold_, indices);

  std::vector<Armor> armors;
  armors.reserve(indices.size());
  for (int idx : indices) {
    armors.push_back(boxToArmor(boxes[idx], class_ids[idx]));
  }
  return armors;
}

Armor YoloDetector::boxToArmor(const cv::Rect & box, int class_id)
{
  // Approximate the two vertical LED lights as the left and right edges of the bbox.
  // A thin RotatedRect on each edge gives the Light struct the top/bottom points
  // that the number extractor and PnP solver expect.
  float light_w = box.width * 0.08f;
  float cx_left = box.x;
  float cx_right = box.x + box.width;
  float cy = box.y + box.height / 2.0f;

  cv::RotatedRect left_rect(cv::Point2f(cx_left, cy), cv::Size2f(light_w, box.height), 0.0f);
  cv::RotatedRect right_rect(cv::Point2f(cx_right, cy), cv::Size2f(light_w, box.height), 0.0f);

  Light left_light(left_rect);
  Light right_light(right_rect);
  left_light.color = -1;
  right_light.color = -1;

  Armor armor(left_light, right_light);
  armor.type = (class_id == 0) ? ArmorType::SMALL : ArmorType::LARGE;
  return armor;
}

}  // namespace sn_auto_aim
