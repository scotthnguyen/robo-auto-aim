// Scott Nguyen — MIT License

#include <gtest/gtest.h>
#include <tf2/LinearMath/Quaternion.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <string>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "armor_tracker/tracker.hpp"

using sn_auto_aim::ArmorsNum;
using sn_auto_aim::ExtendedKalmanFilter;
using sn_auto_aim::Tracker;
using Armors = auto_aim_interfaces::msg::Armors;
using Armor = auto_aim_interfaces::msg::Armor;

namespace
{

constexpr double DT = 0.01;
constexpr double MAX_MATCH_DISTANCE = 0.15;
constexpr double MAX_MATCH_YAW_DIFF = 1.0;
constexpr int TRACKING_THRES = 5;
constexpr int LOST_THRES = 3;

// Mirrors the 9-state circular-motion model that ArmorTrackerNode wires up, so
// these tests exercise the same filter the node runs.
ExtendedKalmanFilter makeTargetEkf()
{
  auto f = [](const Eigen::VectorXd & x) {
    Eigen::VectorXd x_new = x;
    x_new(0) += x(1) * DT;
    x_new(2) += x(3) * DT;
    x_new(4) += x(5) * DT;
    x_new(6) += x(7) * DT;
    return x_new;
  };
  auto j_f = [](const Eigen::VectorXd &) {
    Eigen::MatrixXd m = Eigen::MatrixXd::Identity(9, 9);
    m(0, 1) = m(2, 3) = m(4, 5) = m(6, 7) = DT;
    return m;
  };
  auto h = [](const Eigen::VectorXd & x) {
    Eigen::VectorXd z(4);
    double xc = x(0), yc = x(2), yaw = x(6), r = x(8);
    z(0) = xc - r * cos(yaw);
    z(1) = yc - r * sin(yaw);
    z(2) = x(4);
    z(3) = x(6);
    return z;
  };
  auto j_h = [](const Eigen::VectorXd & x) {
    Eigen::MatrixXd m = Eigen::MatrixXd::Zero(4, 9);
    double yaw = x(6), r = x(8);
    m(0, 0) = 1; m(0, 6) = r * sin(yaw); m(0, 8) = -cos(yaw);
    m(1, 2) = 1; m(1, 6) = -r * cos(yaw); m(1, 8) = -sin(yaw);
    m(2, 4) = 1;
    m(3, 6) = 1;
    return m;
  };
  auto u_q = []() { return Eigen::MatrixXd::Identity(9, 9) * 1e-2; };
  auto u_r = [](const Eigen::VectorXd &) {
    return Eigen::MatrixXd::Identity(4, 4) * 1e-3;
  };
  Eigen::MatrixXd p0 = Eigen::MatrixXd::Identity(9, 9);
  return ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, p0};
}

std::unique_ptr<Tracker> makeTracker()
{
  auto tracker = std::unique_ptr<Tracker>(new Tracker(MAX_MATCH_DISTANCE, MAX_MATCH_YAW_DIFF));
  tracker->ekf = makeTargetEkf();
  tracker->tracking_thres = TRACKING_THRES;
  tracker->lost_thres = LOST_THRES;
  return tracker;
}

Armor makeArmor(
  const std::string & number, double x, double y, double z, double yaw,
  const std::string & type = "small", double distance_to_center = 0.0)
{
  Armor armor;
  armor.number = number;
  armor.type = type;
  armor.distance_to_image_center = distance_to_center;
  armor.pose.position.x = x;
  armor.pose.position.y = y;
  armor.pose.position.z = z;
  tf2::Quaternion q;
  q.setRPY(0, 0, yaw);
  armor.pose.orientation = tf2::toMsg(q);
  return armor;
}

Armors::SharedPtr makeMsg(const std::vector<Armor> & armors)
{
  auto msg = std::make_shared<Armors>();
  msg->armors = armors;
  return msg;
}

// Drives the tracker to TRACKING with a static target at (1, 0, 0), yaw 0.
void driveToTracking(Tracker & tracker)
{
  auto msg = makeMsg({makeArmor("1", 1.0, 0.0, 0.0, 0.0)});
  tracker.init(msg);
  ASSERT_EQ(tracker.tracker_state, Tracker::DETECTING);
  for (int i = 0; i <= TRACKING_THRES; i++) {
    tracker.update(msg);
  }
  ASSERT_EQ(tracker.tracker_state, Tracker::TRACKING);
}

}  // namespace

TEST(TrackerTest, StartsLost)
{
  auto tracker = makeTracker();
  EXPECT_EQ(tracker->tracker_state, Tracker::LOST);
}

TEST(TrackerTest, InitOnEmptyMessageStaysLost)
{
  auto tracker = makeTracker();
  tracker->init(makeMsg({}));
  EXPECT_EQ(tracker->tracker_state, Tracker::LOST);
  EXPECT_EQ(tracker->tracked_id, "");
}

TEST(TrackerTest, InitLocksOntoArmorNearestImageCenter)
{
  auto tracker = makeTracker();
  tracker->init(makeMsg({
    makeArmor("3", 2.0, 0.5, 0.0, 0.0, "small", 300.0),
    makeArmor("7", 1.5, 0.0, 0.0, 0.0, "small", 12.0),
    makeArmor("4", 2.5, -0.5, 0.0, 0.0, "small", 180.0),
  }));

  EXPECT_EQ(tracker->tracker_state, Tracker::DETECTING);
  EXPECT_EQ(tracker->tracked_id, "7");
}

TEST(TrackerTest, DetectingPromotesToTrackingAfterThreshold)
{
  auto tracker = makeTracker();
  auto msg = makeMsg({makeArmor("1", 1.0, 0.0, 0.0, 0.0)});
  tracker->init(msg);

  for (int i = 0; i < TRACKING_THRES; i++) {
    tracker->update(msg);
    EXPECT_EQ(tracker->tracker_state, Tracker::DETECTING) << "promoted early at frame " << i;
  }
  tracker->update(msg);
  EXPECT_EQ(tracker->tracker_state, Tracker::TRACKING);
}

TEST(TrackerTest, DetectingDropsBackToLostOnMiss)
{
  auto tracker = makeTracker();
  tracker->init(makeMsg({makeArmor("1", 1.0, 0.0, 0.0, 0.0)}));

  tracker->update(makeMsg({}));
  EXPECT_EQ(tracker->tracker_state, Tracker::LOST);
}

TEST(TrackerTest, TrackingSurvivesBriefOcclusionThenGivesUp)
{
  auto tracker = makeTracker();
  driveToTracking(*tracker);

  // First miss only demotes to TEMP_LOST — the target may just be occluded.
  tracker->update(makeMsg({}));
  EXPECT_EQ(tracker->tracker_state, Tracker::TEMP_LOST);

  for (int i = 0; i < LOST_THRES; i++) {
    tracker->update(makeMsg({}));
  }
  EXPECT_EQ(tracker->tracker_state, Tracker::LOST);
}

TEST(TrackerTest, TempLostRecoversToTracking)
{
  auto tracker = makeTracker();
  driveToTracking(*tracker);

  tracker->update(makeMsg({}));
  ASSERT_EQ(tracker->tracker_state, Tracker::TEMP_LOST);

  tracker->update(makeMsg({makeArmor("1", 1.0, 0.0, 0.0, 0.0)}));
  EXPECT_EQ(tracker->tracker_state, Tracker::TRACKING);
}

TEST(TrackerTest, IgnoresArmorsWithADifferentNumber)
{
  auto tracker = makeTracker();
  driveToTracking(*tracker);

  // Same position, wrong id — must not be associated.
  tracker->update(makeMsg({makeArmor("2", 1.0, 0.0, 0.0, 0.0)}));
  EXPECT_EQ(tracker->tracker_state, Tracker::TEMP_LOST);
}

TEST(TrackerTest, EstimatedCenterReprojectsOntoObservedArmor)
{
  auto tracker = makeTracker();
  const double xa = 1.0, ya = 0.25, za = -0.1, yaw = 0.3;
  auto msg = makeMsg({makeArmor("1", xa, ya, za, yaw)});
  tracker->init(msg);
  for (int i = 0; i < 20; i++) {
    tracker->update(msg);
  }

  // xc/r are individually weakly observable, but the observation model must map
  // the estimated robot center back onto the armor we actually saw.
  const auto & s = tracker->target_state;
  EXPECT_NEAR(s(0) - s(8) * cos(s(6)), xa, 1e-2);
  EXPECT_NEAR(s(2) - s(8) * sin(s(6)), ya, 1e-2);
  EXPECT_NEAR(s(4), za, 1e-2);
  EXPECT_NEAR(s(6), yaw, 1e-2);
}

TEST(TrackerTest, RadiusStaysWithinPhysicalBounds)
{
  auto tracker = makeTracker();
  // An armor far from where the filter expects it pushes hard on the radius.
  auto msg = makeMsg({makeArmor("1", 1.0, 0.0, 0.0, 0.0)});
  tracker->init(msg);
  for (int i = 0; i < 50; i++) {
    tracker->update(makeMsg({makeArmor("1", 1.0 + 0.002 * i, 0.0, 0.0, 0.0)}));
    EXPECT_GE(tracker->target_state(8), 0.12);
    EXPECT_LE(tracker->target_state(8), 0.4);
  }
}

TEST(TrackerTest, ArmorJumpAdoptsNewYawAndSwapsRadii)
{
  auto tracker = makeTracker();
  driveToTracking(*tracker);

  const double r_before = tracker->target_state(8);
  const double another_r_before = tracker->another_r;
  ASSERT_EQ(tracker->tracked_armors_num, ArmorsNum::NORMAL_4);

  // A 90 deg yaw step on the only same-id armor is a plate switch, not a new
  // target: the robot spun and the next plate rotated into view.
  const double jumped_yaw = M_PI / 2;
  tracker->update(makeMsg({makeArmor("1", 1.0, 0.0, 0.0, jumped_yaw)}));

  EXPECT_NEAR(tracker->target_state(6), jumped_yaw, 1e-6);
  EXPECT_NEAR(tracker->target_state(8), another_r_before, 1e-6);
  EXPECT_NEAR(tracker->another_r, r_before, 1e-6);
}

TEST(TrackerTest, ArmorsNumFollowsIdAndType)
{
  auto balance = makeTracker();
  balance->init(makeMsg({makeArmor("4", 1.0, 0.0, 0.0, 0.0, "large")}));
  EXPECT_EQ(balance->tracked_armors_num, ArmorsNum::BALANCE_2);

  auto outpost = makeTracker();
  outpost->init(makeMsg({makeArmor("outpost", 1.0, 0.0, 0.0, 0.0, "small")}));
  EXPECT_EQ(outpost->tracked_armors_num, ArmorsNum::OUTPOST_3);

  auto normal = makeTracker();
  normal->init(makeMsg({makeArmor("1", 1.0, 0.0, 0.0, 0.0, "small")}));
  EXPECT_EQ(normal->tracked_armors_num, ArmorsNum::NORMAL_4);
}

TEST(TrackerTest, ReacquireAfterLossRestartsDetectCount)
{
  auto tracker = makeTracker();
  driveToTracking(*tracker);
  for (int i = 0; i <= LOST_THRES + 1; i++) {
    tracker->update(makeMsg({}));
  }
  ASSERT_EQ(tracker->tracker_state, Tracker::LOST);

  // A fresh lock must serve the full DETECTING probation again rather than
  // inheriting a stale counter and jumping straight to TRACKING.
  auto msg = makeMsg({makeArmor("2", 1.2, 0.0, 0.0, 0.0)});
  tracker->init(msg);
  EXPECT_EQ(tracker->tracked_id, "2");
  for (int i = 0; i < TRACKING_THRES; i++) {
    tracker->update(msg);
    EXPECT_EQ(tracker->tracker_state, Tracker::DETECTING) << "promoted early at frame " << i;
  }
  tracker->update(msg);
  EXPECT_EQ(tracker->tracker_state, Tracker::TRACKING);
}
