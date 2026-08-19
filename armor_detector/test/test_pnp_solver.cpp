// Scott Nguyen — MIT License

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <vector>

#include "armor_detector/armor.hpp"
#include "armor_detector/pnp_solver.hpp"

using sn_auto_aim::Armor;
using sn_auto_aim::ArmorType;
using sn_auto_aim::PnPSolver;

namespace
{

constexpr double FX = 1000.0;
constexpr double FY = 1000.0;
constexpr double CX = 640.0;
constexpr double CY = 360.0;

// Row-major 3x3 intrinsics, matching sensor_msgs/CameraInfo.k
const std::array<double, 9> K = {FX, 0.0, CX, 0.0, FY, CY, 0.0, 0.0, 1.0};

// Projects a fronto-parallel armor of the given real-world size onto the image
// plane at distance z, then packs the four corners into an Armor.
Armor projectArmor(ArmorType type, double width_m, double height_m, double z)
{
  const double du = FX * (width_m / 2.0) / z;
  const double dv = FY * (height_m / 2.0) / z;

  Armor armor;
  armor.type = type;
  armor.left_light.top = cv::Point2f(CX - du, CY - dv);
  armor.left_light.bottom = cv::Point2f(CX - du, CY + dv);
  armor.right_light.top = cv::Point2f(CX + du, CY - dv);
  armor.right_light.bottom = cv::Point2f(CX + du, CY + dv);
  armor.center = cv::Point2f(CX, CY);
  return armor;
}

std::vector<cv::Point2f> imagePointsOf(const Armor & armor)
{
  return {armor.left_light.bottom, armor.left_light.top, armor.right_light.top,
          armor.right_light.bottom};
}

// Same corner order PnPSolver uses internally, in the model frame
// (x forward, y left, z up).
std::vector<cv::Point3f> objectPointsOf(double width_m, double height_m)
{
  const float hy = static_cast<float>(width_m / 2.0);
  const float hz = static_cast<float>(height_m / 2.0);
  return {{0, hy, -hz}, {0, hy, hz}, {0, -hy, hz}, {0, -hy, -hz}};
}

// Rotation angle of R_a relative to R_b, in radians.
double relativeAngle(const cv::Mat & rvec_a, const cv::Mat & rvec_b)
{
  cv::Mat ra, rb;
  cv::Rodrigues(rvec_a, ra);
  cv::Rodrigues(rvec_b, rb);
  const cv::Mat rel = ra * rb.t();
  const double trace = rel.at<double>(0, 0) + rel.at<double>(1, 1) + rel.at<double>(2, 2);
  return std::acos(std::max(-1.0, std::min(1.0, (trace - 1.0) / 2.0)));
}

}  // namespace

TEST(PnPSolverTest, RecoversDistanceToSmallArmorOnAxis)
{
  PnPSolver solver(K, {0.0, 0.0, 0.0, 0.0, 0.0});
  auto armor = projectArmor(ArmorType::SMALL, 0.135, 0.055, 2.0);

  cv::Mat rvec, tvec;
  ASSERT_TRUE(solver.solvePnP(armor, rvec, tvec));

  EXPECT_NEAR(tvec.at<double>(0), 0.0, 1e-3);
  EXPECT_NEAR(tvec.at<double>(1), 0.0, 1e-3);
  EXPECT_NEAR(tvec.at<double>(2), 2.0, 1e-3);
}

TEST(PnPSolverTest, RecoversDistanceToLargeArmorOnAxis)
{
  PnPSolver solver(K, {0.0, 0.0, 0.0, 0.0, 0.0});
  auto armor = projectArmor(ArmorType::LARGE, 0.225, 0.055, 3.5);

  cv::Mat rvec, tvec;
  ASSERT_TRUE(solver.solvePnP(armor, rvec, tvec));

  EXPECT_NEAR(tvec.at<double>(2), 3.5, 1e-3);
}

TEST(PnPSolverTest, PoseReprojectsOntoTheOriginalCorners)
{
  PnPSolver solver(K, {0.0, 0.0, 0.0, 0.0, 0.0});
  auto armor = projectArmor(ArmorType::SMALL, 0.135, 0.055, 2.0);

  cv::Mat rvec, tvec;
  ASSERT_TRUE(solver.solvePnP(armor, rvec, tvec));

  std::vector<cv::Point2f> reprojected;
  cv::projectPoints(
    objectPointsOf(0.135, 0.055), rvec, tvec, cv::Mat(3, 3, CV_64F, const_cast<double *>(K.data())),
    cv::Mat::zeros(1, 5, CV_64F), reprojected);

  const auto expected = imagePointsOf(armor);
  ASSERT_EQ(reprojected.size(), expected.size());
  for (size_t i = 0; i < expected.size(); i++) {
    EXPECT_NEAR(reprojected[i].x, expected[i].x, 0.1) << "corner " << i;
    EXPECT_NEAR(reprojected[i].y, expected[i].y, 0.1) << "corner " << i;
  }
}

TEST(PnPSolverTest, ArmorSizeIsNotInterchangeable)
{
  PnPSolver solver(K, {0.0, 0.0, 0.0, 0.0, 0.0});
  auto armor = projectArmor(ArmorType::SMALL, 0.135, 0.055, 2.0);

  cv::Mat rvec_small, tvec_small;
  ASSERT_TRUE(solver.solvePnP(armor, rvec_small, tvec_small));

  // Same pixels through the large-armor model. Both plates are 55 mm tall, so
  // the height still pins the range at 2 m; the extra width has to be absorbed
  // by yawing the plate away from the camera until 225 mm foreshortens to the
  // 135 mm we drew. That angle is acos(135/225) — which only comes out right if
  // the modelled armor dimensions are.
  armor.type = ArmorType::LARGE;
  cv::Mat rvec_large, tvec_large;
  ASSERT_TRUE(solver.solvePnP(armor, rvec_large, tvec_large));

  EXPECT_NEAR(tvec_large.at<double>(2), 2.0, 1e-3);
  EXPECT_NEAR(relativeAngle(rvec_large, rvec_small), std::acos(135.0 / 225.0), 1e-2);
}

TEST(PnPSolverTest, OffAxisArmorShiftsInTheExpectedDirection)
{
  PnPSolver solver(K, {0.0, 0.0, 0.0, 0.0, 0.0});
  auto armor = projectArmor(ArmorType::SMALL, 0.135, 0.055, 2.0);
  // Slide the whole plate right and down in the image.
  for (auto * p : {&armor.left_light.top, &armor.left_light.bottom, &armor.right_light.top,
                   &armor.right_light.bottom}) {
    p->x += 100.0f;
    p->y += 50.0f;
  }

  cv::Mat rvec, tvec;
  ASSERT_TRUE(solver.solvePnP(armor, rvec, tvec));

  // OpenCV camera frame: +x right, +y down, +z forward.
  EXPECT_GT(tvec.at<double>(0), 0.0);
  EXPECT_GT(tvec.at<double>(1), 0.0);
  EXPECT_NEAR(tvec.at<double>(2), 2.0, 0.05);
}

TEST(PnPSolverTest, HandlesEmptyDistortionCoefficients)
{
  // An uncalibrated camera publishes CameraInfo with an empty d vector; that
  // must not be read as five doubles.
  PnPSolver solver(K, {});
  auto armor = projectArmor(ArmorType::SMALL, 0.135, 0.055, 2.0);

  cv::Mat rvec, tvec;
  ASSERT_TRUE(solver.solvePnP(armor, rvec, tvec));
  EXPECT_NEAR(tvec.at<double>(2), 2.0, 1e-3);
}

TEST(PnPSolverTest, HandlesEightTermDistortionModel)
{
  // rational_polynomial cameras publish eight coefficients.
  PnPSolver solver(K, std::vector<double>(8, 0.0));
  auto armor = projectArmor(ArmorType::SMALL, 0.135, 0.055, 2.0);

  cv::Mat rvec, tvec;
  ASSERT_TRUE(solver.solvePnP(armor, rvec, tvec));
  EXPECT_NEAR(tvec.at<double>(2), 2.0, 1e-3);
}

TEST(PnPSolverTest, DistanceToCenterIsEuclidean)
{
  PnPSolver solver(K, {0.0, 0.0, 0.0, 0.0, 0.0});

  EXPECT_NEAR(solver.calculateDistanceToCenter(cv::Point2f(CX, CY)), 0.0f, 1e-4);
  EXPECT_NEAR(solver.calculateDistanceToCenter(cv::Point2f(CX + 3, CY + 4)), 5.0f, 1e-4);
  EXPECT_NEAR(solver.calculateDistanceToCenter(cv::Point2f(CX - 3, CY - 4)), 5.0f, 1e-4);
}
