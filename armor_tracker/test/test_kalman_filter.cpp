#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

#include "armor_tracker/extended_kalman_filter.hpp"

using sn_auto_aim::ExtendedKalmanFilter;

// 2D constant-velocity model: state=[x, vx], measurement=[x]
static constexpr double DT = 0.01;

static ExtendedKalmanFilter make_cv_ekf()
{
  auto f = [](const Eigen::VectorXd & x) {
    Eigen::VectorXd xn = x;
    xn(0) += x(1) * DT;
    return xn;
  };
  auto j_f = [](const Eigen::VectorXd &) {
    Eigen::MatrixXd F(2, 2);
    F << 1, DT, 0, 1;
    return F;
  };
  auto h = [](const Eigen::VectorXd & x) {
    Eigen::VectorXd z(1);
    z(0) = x(0);
    return z;
  };
  auto j_h = [](const Eigen::VectorXd &) {
    Eigen::MatrixXd H(1, 2);
    H << 1, 0;
    return H;
  };
  auto u_q = []() { return Eigen::MatrixXd::Identity(2, 2) * 0.1; };
  auto u_r = [](const Eigen::VectorXd &) {
    return Eigen::DiagonalMatrix<double, 1>(0.01);
  };
  Eigen::MatrixXd P0 = Eigen::MatrixXd::Identity(2, 2);
  return ExtendedKalmanFilter{f, h, j_f, j_h, u_q, u_r, P0};
}

TEST(EkfTest, PredictAdvancesPositionByVelocity)
{
  auto ekf = make_cv_ekf();
  Eigen::VectorXd x0(2);
  x0 << 5.0, 2.0;
  ekf.setState(x0);

  auto x_pred = ekf.predict();
  EXPECT_NEAR(x_pred(0), 5.0 + 2.0 * DT, 1e-9);
  EXPECT_NEAR(x_pred(1), 2.0, 1e-9);
}

TEST(EkfTest, UpdatePullsStateTowardMeasurement)
{
  auto ekf = make_cv_ekf();
  Eigen::VectorXd x0(2);
  x0 << 10.0, 0.0;
  ekf.setState(x0);

  // Single predict + update with measurement at 0 — state should move toward 0
  ekf.predict();
  Eigen::VectorXd z(1);
  z(0) = 0.0;
  auto x_post = ekf.update(z);

  EXPECT_LT(x_post(0), 10.0);
}

TEST(EkfTest, RepeatedUpdatesConvergeOnTruePosition)
{
  auto ekf = make_cv_ekf();
  Eigen::VectorXd x0(2);
  x0 << 20.0, 0.0;
  ekf.setState(x0);

  Eigen::VectorXd z(1);
  z(0) = 0.0;
  for (int i = 0; i < 200; i++) {
    ekf.predict();
    ekf.update(z);
  }

  auto x_final = ekf.predict();
  EXPECT_NEAR(x_final(0), 0.0, 0.5);
}

TEST(EkfTest, SetStateOverridesPosteriori)
{
  auto ekf = make_cv_ekf();
  Eigen::VectorXd x0(2);
  x0 << 0.0, 0.0;
  ekf.setState(x0);
  ekf.predict();

  Eigen::VectorXd x_reset(2);
  x_reset << 99.0, -3.0;
  ekf.setState(x_reset);

  auto x_pred = ekf.predict();
  EXPECT_NEAR(x_pred(0), 99.0 + (-3.0) * DT, 1e-9);
}
