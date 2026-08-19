// Scott Nguyen — MIT License

#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "armor_detector/detector.hpp"
#include "armor_detector/yolo_detector.hpp"

using sn_auto_aim::YoloDetector;

namespace
{

constexpr int IMG_W = 320;
constexpr int IMG_H = 240;

const cv::Vec3b RED_LIT(255, 170, 170);
const cv::Vec3b BLUE_LIT(170, 170, 255);
const cv::Vec3b DIM(40, 40, 40);

// An rgb8 scene with a plate spanning `box`, lit strips down both edges.
cv::Mat sceneWithPlate(const cv::Rect & box, const cv::Vec3b & strip_rgb)
{
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);
  const int strip_w = std::max(2, box.width / 12);
  img(cv::Rect(box.x, box.y, strip_w, box.height)).setTo(strip_rgb);
  img(cv::Rect(box.x + box.width - strip_w, box.y, strip_w, box.height)).setTo(strip_rgb);
  return img;
}

cv::Mat loadRgb(const std::string & name)
{
  cv::Mat bgr = cv::imread(std::string(TEST_IMAGE_DIR) + "/" + name, cv::IMREAD_COLOR);
  cv::Mat rgb;
  if (!bgr.empty()) cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

}  // namespace

TEST(YoloColorTest, ReadsRedStrips)
{
  const cv::Rect box(100, 80, 120, 60);
  EXPECT_EQ(YoloDetector::sampleLightColor(sceneWithPlate(box, RED_LIT), box), sn_auto_aim::RED);
}

TEST(YoloColorTest, ReadsBlueStrips)
{
  const cv::Rect box(100, 80, 120, 60);
  EXPECT_EQ(YoloDetector::sampleLightColor(sceneWithPlate(box, BLUE_LIT), box), sn_auto_aim::BLUE);
}

TEST(YoloColorTest, ReportsUnknownWhenNothingIsLit)
{
  const cv::Rect box(100, 80, 120, 60);
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);
  img(box).setTo(DIM);

  // A plate with no lit strips must not be guessed at — an unknown colour is
  // dropped downstream, which is the safe outcome.
  EXPECT_EQ(YoloDetector::sampleLightColor(img, box), -1);
}

TEST(YoloColorTest, PlateInteriorDoesNotOutvoteTheStrips)
{
  // A big red surface behind/inside a blue-lit plate: sampling the whole box
  // would call this red. Only the edge bands may vote.
  const cv::Rect box(100, 80, 120, 60);
  cv::Mat img = sceneWithPlate(box, BLUE_LIT);
  img(cv::Rect(box.x + box.width / 4, box.y, box.width / 2, box.height)).setTo(RED_LIT);

  EXPECT_EQ(YoloDetector::sampleLightColor(img, box), sn_auto_aim::BLUE);
}

TEST(YoloColorTest, HandlesBoxRunningOffTheImageEdge)
{
  // YOLO boxes are unscaled from letterbox space and can extend past the frame.
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);
  const cv::Rect visible(0, 100, 60, 50);
  img(visible).setTo(RED_LIT);

  const cv::Rect box(-40, 100, 100, 50);
  EXPECT_EQ(YoloDetector::sampleLightColor(img, box), sn_auto_aim::RED);
}

TEST(YoloColorTest, FullyOffscreenBoxIsUnknownRatherThanACrash)
{
  cv::Mat img = cv::Mat::zeros(IMG_H, IMG_W, CV_8UC3);
  EXPECT_EQ(YoloDetector::sampleLightColor(img, cv::Rect(-500, -500, 100, 50)), -1);
  EXPECT_EQ(YoloDetector::sampleLightColor(img, cv::Rect(IMG_W + 10, 10, 100, 50)), -1);
}

// The synthetic cases above use clean primaries; these run the same code over
// real captures, where the strips have saturated near-white cores.
TEST(YoloColorTest, ReadsRedFromARealCapture)
{
  cv::Mat img = loadRgb("red.png");
  ASSERT_FALSE(img.empty()) << "missing docs/red.png";
  // Box spanning both strips (centers at x~40 and x~286, ~99 px tall).
  EXPECT_EQ(YoloDetector::sampleLightColor(img, cv::Rect(33, 42, 260, 99)), sn_auto_aim::RED);
}

TEST(YoloColorTest, ReadsBlueFromARealCapture)
{
  cv::Mat img = loadRgb("blue.png");
  ASSERT_FALSE(img.empty()) << "missing docs/blue.png";
  // Box spanning both strips (centers at x~37 and x~284, ~100 px tall).
  EXPECT_EQ(YoloDetector::sampleLightColor(img, cv::Rect(30, 48, 261, 100)), sn_auto_aim::BLUE);
}

TEST(YoloColorTest, ReadsRedFromAFullSceneCapture)
{
  cv::Mat img = loadRgb("raw.png");
  ASSERT_FALSE(img.empty()) << "missing docs/raw.png";
  // Strips at x~54 and x~277, ~90 px tall, over a lit background.
  EXPECT_EQ(YoloDetector::sampleLightColor(img, cv::Rect(45, 50, 242, 92)), sn_auto_aim::RED);
}

// filterByColor is what actually stops the YOLO path from engaging a teammate,
// since the model itself only classifies plate size.
namespace
{
sn_auto_aim::Armor armorWithColor(int color)
{
  sn_auto_aim::Armor armor;
  armor.left_light.color = color;
  armor.right_light.color = color;
  return armor;
}
}  // namespace

TEST(YoloColorTest, FilterKeepsOnlyTheEnemyColor)
{
  std::vector<sn_auto_aim::Armor> armors = {
    armorWithColor(sn_auto_aim::RED), armorWithColor(sn_auto_aim::BLUE),
    armorWithColor(sn_auto_aim::RED)};

  sn_auto_aim::Detector::filterByColor(armors, sn_auto_aim::BLUE);

  ASSERT_EQ(armors.size(), 1u);
  EXPECT_EQ(armors[0].left_light.color, sn_auto_aim::BLUE);
}

TEST(YoloColorTest, FilterDropsPlatesOfOurOwnColor)
{
  std::vector<sn_auto_aim::Armor> armors = {
    armorWithColor(sn_auto_aim::RED), armorWithColor(sn_auto_aim::RED)};

  sn_auto_aim::Detector::filterByColor(armors, sn_auto_aim::BLUE);
  EXPECT_TRUE(armors.empty()) << "a blue-team robot must not engage red plates";
}

TEST(YoloColorTest, FilterDropsUnknownColor)
{
  // -1 comes back when no strip was lit enough to call. Refusing to shoot beats
  // guessing wrong about which team the plate belongs to.
  std::vector<sn_auto_aim::Armor> armors = {armorWithColor(-1)};

  sn_auto_aim::Detector::filterByColor(armors, sn_auto_aim::RED);
  EXPECT_TRUE(armors.empty());

  armors = {armorWithColor(-1)};
  sn_auto_aim::Detector::filterByColor(armors, sn_auto_aim::BLUE);
  EXPECT_TRUE(armors.empty());
}

TEST(YoloColorTest, FilterOnAnEmptyListIsANoop)
{
  std::vector<sn_auto_aim::Armor> armors;
  sn_auto_aim::Detector::filterByColor(armors, sn_auto_aim::RED);
  EXPECT_TRUE(armors.empty());
}
