// Scott Nguyen — MIT License
//
// Regression coverage for the classical pipeline over the real captures kept in
// docs/. The synthetic scenes in test_detector.cpp pin down the geometry rules;
// these pin down behaviour on actual camera frames, where the strips have
// saturated cores, coloured bleed, and a noisy background.

#include <gtest/gtest.h>

#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "armor_detector/detector.hpp"

using sn_auto_aim::ArmorType;
using sn_auto_aim::Detector;

namespace
{

// The shipped defaults from armor_detector.yaml.
constexpr int DEFAULT_BINARY_THRES = 160;

Detector makeDetector(int detect_color, int binary_thres = DEFAULT_BINARY_THRES)
{
  return Detector(
    binary_thres, detect_color, Detector::LightParams{0.1, 0.4, 40.0},
    Detector::ArmorParams{0.7, 0.8, 3.2, 3.2, 5.5, 35.0});
}

// docs/ holds BGR PNGs; the node feeds the detector rgb8, so convert.
cv::Mat loadRgb(const std::string & name)
{
  cv::Mat bgr = cv::imread(std::string(TEST_IMAGE_DIR) + "/" + name, cv::IMREAD_COLOR);
  cv::Mat rgb;
  if (!bgr.empty()) cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

// findLights returns contours in whatever order OpenCV walked them.
std::vector<sn_auto_aim::Light> sortedByX(std::vector<sn_auto_aim::Light> lights)
{
  std::sort(lights.begin(), lights.end(), [](const auto & a, const auto & b) {
    return a.center.x < b.center.x;
  });
  return lights;
}

}  // namespace

TEST(DetectorImageTest, FindsTheArmorInARealRedCapture)
{
  cv::Mat img = loadRgb("raw.png");
  ASSERT_FALSE(img.empty()) << "missing docs/raw.png";

  auto detector = makeDetector(sn_auto_aim::RED);
  auto lights = sortedByX(detector.findLights(img, detector.preprocessImage(img)));

  ASSERT_EQ(lights.size(), 2u);
  EXPECT_NEAR(lights[0].center.x, 54.0f, 6.0f);
  EXPECT_NEAR(lights[1].center.x, 277.0f, 6.0f);
  for (const auto & light : lights) {
    EXPECT_EQ(light.color, sn_auto_aim::RED);
    EXPECT_NEAR(light.length, 90.0, 12.0);
    EXPECT_LT(light.tilt_angle, 10.0f);
  }

  auto armors = detector.matchLights(lights);
  ASSERT_EQ(armors.size(), 1u);
  EXPECT_EQ(armors[0].type, ArmorType::SMALL);
  EXPECT_NEAR(armors[0].center.x, 165.0f, 8.0f);
  EXPECT_NEAR(armors[0].center.y, 96.0f, 8.0f);
}

TEST(DetectorImageTest, IsolatedRedStripsPairIntoOneArmor)
{
  cv::Mat img = loadRgb("red.png");
  ASSERT_FALSE(img.empty()) << "missing docs/red.png";

  auto detector = makeDetector(sn_auto_aim::RED);
  auto lights = detector.findLights(img, detector.preprocessImage(img));

  ASSERT_EQ(lights.size(), 2u);
  for (const auto & light : lights) {
    EXPECT_EQ(light.color, sn_auto_aim::RED);
  }
  EXPECT_EQ(detector.matchLights(lights).size(), 1u);
}

TEST(DetectorImageTest, IsolatedBlueStripsPairIntoOneArmor)
{
  cv::Mat img = loadRgb("blue.png");
  ASSERT_FALSE(img.empty()) << "missing docs/blue.png";

  auto detector = makeDetector(sn_auto_aim::BLUE);
  auto lights = detector.findLights(img, detector.preprocessImage(img));

  ASSERT_EQ(lights.size(), 2u);
  for (const auto & light : lights) {
    EXPECT_EQ(light.color, sn_auto_aim::BLUE);
  }
  EXPECT_EQ(detector.matchLights(lights).size(), 1u);
}

TEST(DetectorImageTest, TeamColorGatingHoldsOnRealCaptures)
{
  cv::Mat red_img = loadRgb("red.png");
  cv::Mat blue_img = loadRgb("blue.png");
  ASSERT_FALSE(red_img.empty());
  ASSERT_FALSE(blue_img.empty());

  // A detector hunting blue must ignore a red plate, and vice versa.
  auto blue_hunter = makeDetector(sn_auto_aim::BLUE);
  EXPECT_TRUE(
    blue_hunter.matchLights(blue_hunter.findLights(red_img, blue_hunter.preprocessImage(red_img)))
      .empty());

  auto red_hunter = makeDetector(sn_auto_aim::RED);
  EXPECT_TRUE(
    red_hunter.matchLights(red_hunter.findLights(blue_img, red_hunter.preprocessImage(blue_img)))
      .empty());
}

TEST(DetectorImageTest, DetectionIsStableAcrossTheUsableThresholdRange)
{
  cv::Mat img = loadRgb("raw.png");
  ASSERT_FALSE(img.empty());

  // The plate should not depend on hitting binary_thres exactly right.
  for (int thres : {60, 100, 130, 160, 200}) {
    auto detector = makeDetector(sn_auto_aim::RED, thres);
    auto lights = detector.findLights(img, detector.preprocessImage(img));
    auto armors = detector.matchLights(lights);
    EXPECT_EQ(lights.size(), 2u) << "binary_thres=" << thres;
    EXPECT_EQ(armors.size(), 1u) << "binary_thres=" << thres;
  }
}
