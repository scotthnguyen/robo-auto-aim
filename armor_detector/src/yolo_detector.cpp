// Scott Nguyen — MIT License

#include "armor_detector/yolo_detector.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>

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
  if (outputs.empty()) {
    return {};
  }

  return postprocess(outputs[0], img, scale, offset);
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
  const cv::Mat & output, const cv::Mat & img, float scale, const cv::Point & offset) const
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
    armors.push_back(
      boxToArmor(boxes[idx], class_ids[idx], sampleLightColor(img, boxes[idx])));
  }
  return armors;
}

int YoloDetector::sampleLightColor(const cv::Mat & img, const cv::Rect & box)
{
  // The LED strips run down the left and right edges of the plate, so sample a
  // narrow band over each and let the brighter channel decide. Only lit pixels
  // get a vote, which keeps the dark plate face and whatever is behind the
  // robot from outvoting the strips themselves.
  const cv::Rect bounds(0, 0, img.cols, img.rows);
  const int band_w = std::max(2, static_cast<int>(box.width * 0.15f));

  int64_t sum_r = 0, sum_b = 0;
  for (int edge = 0; edge < 2; edge++) {
    const int x = edge == 0 ? box.x : box.x + box.width - band_w;
    const cv::Rect band = cv::Rect(x, box.y, band_w, box.height) & bounds;
    if (band.width <= 0 || band.height <= 0) continue;

    const cv::Mat roi = img(band);
    for (int i = 0; i < roi.rows; i++) {
      for (int j = 0; j < roi.cols; j++) {
        // rgb8 input: channel 0 is red, channel 2 is blue.
        const cv::Vec3b & px = roi.at<cv::Vec3b>(i, j);
        if (std::max(px[0], px[2]) < LIGHT_MIN_INTENSITY) continue;
        sum_r += px[0];
        sum_b += px[2];
      }
    }
  }

  // Nothing lit, or a dead heat — refuse to guess.
  if (sum_r == sum_b) return -1;
  return sum_r > sum_b ? RED : BLUE;
}

Armor YoloDetector::boxToArmor(const cv::Rect & box, int class_id, int color)
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
  left_light.color = color;
  right_light.color = color;

  Armor armor(left_light, right_light);
  armor.type = (class_id == 0) ? ArmorType::SMALL : ArmorType::LARGE;
  return armor;
}

}  // namespace sn_auto_aim
