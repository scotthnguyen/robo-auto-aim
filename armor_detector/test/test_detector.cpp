// Scott Nguyen — MIT License

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

#include "armor_detector/detector.hpp"

using sn_auto_aim::ArmorType;
using sn_auto_aim::Detector;

namespace
{

constexpr int IMG_W = 640;
constexpr int IMG_H = 480;
constexpr int BINARY_THRES = 160;
constexpr int LIGHT_HALF_H = 25;  // light strips are ~50 px tall
constexpr int LIGHT_HALF_W = 5;   // ...and ~10 px wide

Detector::LightParams defaultLightParams()
{
  return Detector::LightParams{0.1, 0.4, 40.0};
}

Detector::ArmorParams defaultArmorParams()
{
  return Detector::ArmorParams{0.7, 0.8, 3.2, 3.2, 5.5, 35.0};
}

Detector makeDetector(int detect_color)
{
  return Detector(BINARY_THRES, detect_color, defaultLightParams(), defaultArmorParams());
}

// Draws a lit strip as a filled ellipse. Ellipses (rather than rectangles) keep
// the contour above the 5-point minimum findLights() requires, and are closer
// to the rounded shape of a real LED strip.
void drawLight(cv::Mat & img, int cx, int cy, int color, double angle_deg = 0.0)
{
  // Saturated core with a colored tint, bright enough to clear the binary
  // threshold once converted to grayscale.
  const cv::Scalar rgb = color == sn_auto_aim::RED ? cv::Scalar(255, 180, 180)
                                                   : cv::Scalar(180, 180, 255);
  cv::ellipse(
    img, cv::Point(cx, cy), cv::Size(LIGHT_HALF_W, LIGHT_HALF_H), angle_deg, 0, 360, rgb, -1);
}

// An RGB8 scene (matching the encoding the node feeds in) with a light pair
// whose centers are `separation` px apart.
cv::Mat makeArmorScene(int separation, int color = sn_auto_aim::RED)
{
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);
  drawLight(img, IMG_W / 2 - separation / 2, IMG_H / 2, color);
  drawLight(img, IMG_W / 2 + separation / 2, IMG_H / 2, color);
  return img;
}

}  // namespace

TEST(DetectorTest, PreprocessKeepsLitPixelsAndDropsBackground)
{
  auto detector = makeDetector(sn_auto_aim::RED);
  auto img = makeArmorScene(80);

  auto binary = detector.preprocessImage(img);

  ASSERT_EQ(binary.type(), CV_8UC1);
  ASSERT_EQ(binary.size(), img.size());
  EXPECT_EQ(binary.at<uchar>(0, 0), 0) << "background should threshold to black";
  EXPECT_EQ(binary.at<uchar>(IMG_H / 2, IMG_W / 2 - 40), 255) << "light core should survive";
}

TEST(DetectorTest, FindsBothLightStripsAndTheirColor)
{
  auto detector = makeDetector(sn_auto_aim::RED);
  auto img = makeArmorScene(80);

  auto lights = detector.findLights(img, detector.preprocessImage(img));

  ASSERT_EQ(lights.size(), 2u);
  for (const auto & light : lights) {
    EXPECT_EQ(light.color, sn_auto_aim::RED);
    EXPECT_NEAR(light.length, 2.0 * LIGHT_HALF_H, 4.0);
    EXPECT_NEAR(light.width, 2.0 * LIGHT_HALF_W, 4.0);
    EXPECT_LT(light.tilt_angle, 5.0f);
  }
}

TEST(DetectorTest, ClassifiesCloseLightPairAsSmallArmor)
{
  auto detector = makeDetector(sn_auto_aim::RED);
  // 80 px apart over ~50 px lights => ~1.6 light-lengths, inside the small band.
  auto img = makeArmorScene(80);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  auto armors = detector.matchLights(lights);

  ASSERT_EQ(armors.size(), 1u);
  EXPECT_EQ(armors[0].type, ArmorType::SMALL);
  EXPECT_NEAR(armors[0].center.x, IMG_W / 2, 2.0);
  EXPECT_NEAR(armors[0].center.y, IMG_H / 2, 2.0);
  EXPECT_LT(armors[0].left_light.center.x, armors[0].right_light.center.x);
}

TEST(DetectorTest, ClassifiesWideLightPairAsLargeArmor)
{
  auto detector = makeDetector(sn_auto_aim::RED);
  // 200 px apart over ~50 px lights => ~4 light-lengths, inside the large band.
  auto img = makeArmorScene(200);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  auto armors = detector.matchLights(lights);

  ASSERT_EQ(armors.size(), 1u);
  EXPECT_EQ(armors[0].type, ArmorType::LARGE);
}

TEST(DetectorTest, RejectsLightPairBeyondTheLargeArmorSpacing)
{
  auto detector = makeDetector(sn_auto_aim::RED);
  // ~8 light-lengths apart — too far to be one plate.
  auto img = makeArmorScene(400);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  ASSERT_EQ(lights.size(), 2u);
  EXPECT_TRUE(detector.matchLights(lights).empty());
}

TEST(DetectorTest, IgnoresLightsOfTheEnemyColor)
{
  auto detector = makeDetector(sn_auto_aim::BLUE);
  auto img = makeArmorScene(80, sn_auto_aim::RED);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  ASSERT_EQ(lights.size(), 2u);
  for (const auto & light : lights) {
    EXPECT_EQ(light.color, sn_auto_aim::RED);
  }
  EXPECT_TRUE(detector.matchLights(lights).empty()) << "a blue detector must skip red lights";
}

TEST(DetectorTest, DetectsBlueLightsWhenConfiguredForBlue)
{
  auto detector = makeDetector(sn_auto_aim::BLUE);
  auto img = makeArmorScene(80, sn_auto_aim::BLUE);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  ASSERT_EQ(lights.size(), 2u);
  for (const auto & light : lights) {
    EXPECT_EQ(light.color, sn_auto_aim::BLUE);
  }
  EXPECT_EQ(detector.matchLights(lights).size(), 1u);
}

TEST(DetectorTest, RejectsPairWithAThirdLightBetweenThem)
{
  auto detector = makeDetector(sn_auto_aim::RED);
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);
  drawLight(img, IMG_W / 2 - 100, IMG_H / 2, sn_auto_aim::RED);
  drawLight(img, IMG_W / 2, IMG_H / 2, sn_auto_aim::RED);
  drawLight(img, IMG_W / 2 + 100, IMG_H / 2, sn_auto_aim::RED);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  ASSERT_EQ(lights.size(), 3u);
  auto armors = detector.matchLights(lights);

  // The outer pair straddles the middle light and must be discarded; the two
  // adjacent pairs are 100 px / ~50 px = ~2 light-lengths, so both are valid.
  for (const auto & armor : armors) {
    EXPECT_LT(std::abs(armor.left_light.center.x - armor.right_light.center.x), 150.0f);
  }
}

TEST(DetectorTest, EmptyImageYieldsNoDetections)
{
  auto detector = makeDetector(sn_auto_aim::RED);
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  EXPECT_TRUE(lights.empty());
  EXPECT_TRUE(detector.matchLights(lights).empty());
}

TEST(DetectorTest, VerticallyAlignedLightsDoNotProduceANaNAngle)
{
  // dx == 0 used to divide by zero when measuring the pair's tilt.
  auto detector = makeDetector(sn_auto_aim::RED);
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);
  drawLight(img, IMG_W / 2, IMG_H / 2 - 60, sn_auto_aim::RED);
  drawLight(img, IMG_W / 2, IMG_H / 2 + 60, sn_auto_aim::RED);

  auto lights = detector.findLights(img, detector.preprocessImage(img));
  ASSERT_EQ(lights.size(), 2u);
  EXPECT_TRUE(detector.matchLights(lights).empty());

  ASSERT_FALSE(detector.debug_armors.data.empty());
  for (const auto & d : detector.debug_armors.data) {
    EXPECT_FALSE(std::isnan(d.angle));
    EXPECT_NEAR(d.angle, 90.0, 1e-3);
  }
}
