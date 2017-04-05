#include "ukf.h"
#include <cassert>
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF():
    is_initialized_(false),
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_(true),
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_(true),
    // initial state vector
    n_x_(5),
    n_aug_(7),
    x_(n_x_),
    // initial covariance matrix
    P_(n_x_, n_x_),
    // initial Xsig_pred_
    Xsig_pred_(n_x_, 2 * n_aug_ + 1),
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_(2.5),  // TODO: Tune this
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_(.5),  // TODO: Tune this
    // Laser measurement noise standard deviation position1 in m
    std_laspx_(0.15),
    // Laser measurement noise standard deviation position2 in m
    std_laspy_(0.15),
    // Radar measurement noise standard deviation radius in m
    std_radr_(0.3),
    // Radar measurement noise standard deviation angle in rad
    std_radphi_(0.03),
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_(0.3),
    // Lamda coefficient for unscented transform
    lambda_(3-n_aug_),
    H_laser_(2, 5),
    R_laser_(2, 2){
  weights_ = VectorXd::Constant(2*n_aug_+1, 0.5/(lambda_ + n_aug_));
  weights_(0) *= 2 * lambda_;
  x_.fill(0);

  Xsig_pred_.fill(0);

  P_ = MatrixXd::Identity(n_x_, n_x_);

  H_laser_ << 1, 0, 0, 0, 0,
              0, 1, 0, 0, 0;

  R_laser_ << std_laspx_ * std_laspx_, 0,
              0, std_laspy_ * std_laspy_;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
    // Process Lidar
    if (! is_initialized_) {
      // Initialize mean and covariance
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      P_(0, 0) = R_laser_(0, 0);
      P_(1, 1) = R_laser_(1, 1);
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
      return;
    }
    else if (use_laser_) {
      double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
      Predict(dt);
      UpdateLidar(meas_package);
      time_us_ = meas_package.timestamp_;
    }
  }
  else {
    // Process Radar
    if (! is_initialized_) {
      // Initialize mean
      const double& rho = meas_package.raw_measurements_(0);
      const double& phi = meas_package.raw_measurements_(1);
      const double& rho_dot = meas_package.raw_measurements_(2);
      x_(0) = rho * cos(phi);
      x_(1) = rho * sin(phi);
      x_(2) = rho_dot;
      // Initialize covariance using Jacobian
      const int n_z = meas_package.raw_measurements_.size();
      MatrixXd R_radar_ = MatrixXd::Zero(n_z, n_z);
      R_radar_(0, 0) = std_radr_ * std_radr_;
      R_radar_(1, 1) = std_radphi_ * std_radphi_;
      R_radar_(2, 2) = std_radrd_ * std_radrd_;
      MatrixXd H_inv = HInvjRadar(meas_package.raw_measurements_);
      P_.topLeftCorner(3, 3) = (H_inv * R_radar_ * H_inv.transpose()).topLeftCorner(3, 3);
      time_us_ = meas_package.timestamp_;
      is_initialized_ = true;
      return;
    }
    else if (use_radar_)
    {
      // Perform filter prediction/update from radar
      double dt = (meas_package.timestamp_ - time_us_) / 1000000.0;
      Predict(dt);
      UpdateRadar(meas_package);
      time_us_ = meas_package.timestamp_;
    }
  }
}

MatrixXd UKF::HInvjRadar(const Eigen::VectorXd &z_measurement) {
  double r = z_measurement(0);
  double phi = z_measurement(1);
  double r_dot = z_measurement(2);
  double s = sin(phi);
  double c = sin(phi);
  return (MatrixXd(5,3) << c, -r * s, 0,
                           s, r * c, 0,
                           0, 0, 1,
                           0, 1, 0,
                           0, 0, 0).finished();
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */

void UKF::Predict(double dt) {
  assert(x_.allFinite());
  assert(P_.allFinite());
  // P Matrix must be positive semi-definite
  assert(P_.llt().info() != Eigen::NumericalIssue);

  // Create x_aug Vector
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug.head(n_x_) = x_;
  assert(x_aug.allFinite());

  // Create P_aug Matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_ + 1, n_x_+1) = std_yawdd_ * std_yawdd_;
  assert(P_aug.allFinite());
  // P_aug Matrix must be positive semi-definite
  assert(P_aug.llt().info() != Eigen::NumericalIssue);

  // Create Xsig_aug Matrix
  MatrixXd A = P_aug.llt().matrixL();
  A *= sqrt(lambda_ + n_aug_);
  MatrixXd Xsig_aug = x_aug.replicate(1, 2 * n_aug_ + 1);
  Xsig_aug.block(0, 1, n_aug_, n_aug_) += A;
  Xsig_aug.rightCols(n_aug_) -= A;
  assert(Xsig_aug.allFinite());

  // Predict XsigPred
  double dt2 = dt * dt;
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    const double& vk = Xsig_aug(2, i);
    const double& psi_k = Xsig_aug(3, i);
    const double& psi_dot_k = Xsig_aug(4, i);
    const double& nu_a_k = Xsig_aug(5, i);
    const double& nu_psi_ddot_k = Xsig_aug(6, i);
    Xsig_pred_.col(i) = Xsig_aug.block(0, i, n_x_, 1);
    if (abs(psi_dot_k) < .000001) {
      Xsig_pred_(0, i) += (vk + .5 * dt * nu_a_k) * cos(psi_k) * dt;
      Xsig_pred_(1, i) += (vk + .5 * dt * nu_a_k) * sin(psi_k) * dt;
      Xsig_pred_(2, i) += dt * nu_a_k;
      Xsig_pred_(3, i) += psi_dot_k * dt + .5 * dt2 * nu_psi_ddot_k;
      Xsig_pred_(4, i) += dt * nu_psi_ddot_k;
    }
    else {
      const double& sin_psi = sin(psi_k);
      const double& cos_psi = cos(psi_k);
      const double& psi_k1 = psi_k + psi_dot_k * dt;
      Xsig_pred_(0, i) += vk/psi_dot_k * (sin(psi_k1) - sin_psi) + .5 * dt2 * cos_psi * nu_a_k;
      Xsig_pred_(1, i) += vk/psi_dot_k * (-cos(psi_k1) + cos_psi) + .5 * dt2 * sin_psi * nu_a_k;
      Xsig_pred_(2, i) += dt * nu_a_k;
      Xsig_pred_(3, i)  = psi_k1 + .5 * dt2 * nu_psi_ddot_k;
      Xsig_pred_(4, i) += dt * nu_psi_ddot_k;
    }
  }
  assert(Xsig_pred_.allFinite());

  // Calculate Updated State Vector x_
  x_ = Xsig_pred_ * weights_;
  x_(3) = atan2(sin(x_(3)), cos(x_(3)));

  // Calculate Updated Covariance matrix P_
  P_ = MatrixXd::Zero(n_x_, n_x_);
  for (int i = 0; i < Xsig_pred_.cols(); ++i) {
    VectorXd d = Xsig_pred_.col(i) - x_;
    d(3) = atan2(sin(d(3)), cos(d(3)));
    P_ += weights_(i) * d * d.transpose();
  }

  cout << "Predict(" << dt << ")" <<endl
       << "X_aug:" << endl << x_aug <<endl
       << "P_aug:" << endl << P_aug << endl
       << "A:" << endl << A << endl
       << "Xsig_aug:" << endl << Xsig_aug << endl
       << "Xsig_pred:" << endl << Xsig_pred_  << endl
       << "x:" << endl << x_ << endl
       << "P:" << endl << P_ << endl;

  assert(x_.allFinite());
  assert(P_.allFinite());
  assert(P_.llt().info() != Eigen::NumericalIssue);
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  assert(x_.allFinite());
  assert(P_.allFinite());
  assert(meas_package.raw_measurements_.allFinite());
  assert(meas_package.raw_measurements_.size() == 2);

  // This is a linear update
  VectorXd y = meas_package.raw_measurements_ - H_laser_ * x_;
  MatrixXd S = H_laser_ * P_ * H_laser_.transpose() + R_laser_;
  MatrixXd S_inv = S.inverse();
  MatrixXd K = P_ * H_laser_.transpose() * S.inverse();
  x_ = x_ + K * y;
  x_(3) = atan2(sin(x_(3)), cos(x_(3)));
  P_ = (MatrixXd::Identity(x_.size(), x_.size()) - K * H_laser_) * P_;

  // Calculate the NIS
  NIS_laser_ = y.transpose() * S_inv * y;

  cout << "UpdateLidar(" << meas_package.raw_measurements_.transpose() << ")" << endl
       << "y:" << endl << y << endl
       << "H:" << endl << H_laser_ << endl
       << "R:" << endl << R_laser_ << endl
       << "S:" << endl << S << endl
       << "K:" << endl << K << endl
       << "KH:" << endl << K*H_laser_ << endl
       << "x:" << endl << x_ << endl
       << "P:" <<endl << P_ << endl;

  assert(x_.allFinite());
  assert(P_.allFinite());
  assert(P_.llt().info() != Eigen::NumericalIssue);
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  assert(x_.allFinite());
  assert(P_.allFinite());
  assert(meas_package.raw_measurements_.allFinite());
  assert(meas_package.raw_measurements_.size() == 3);

  const int n_z = meas_package.raw_measurements_.size();

  // Calculate the Z_sig Matrix
  MatrixXd Zsig(n_z, 2 * n_aug_ + 1);
  for (int i = 0; i < Zsig.cols(); ++i)
  {
      const double& px = Xsig_pred_(0, i);
      const double& py = Xsig_pred_(1, i);
      const double& v = Xsig_pred_(2, i);
      const double& psi = Xsig_pred_(3, i);
      Zsig(0, i) = sqrt(px * px + py * py);
      Zsig(1, i) = atan2(py, px);
      if (Zsig(0, i) < .000001) {
        Zsig(2, i) = 0;
      }
      else {
        Zsig(2, i) = (px * cos(psi) * v + py * sin(psi) * v) / Zsig(0, i);
      }
  }

  // Calculate Predicted Measurement Vector
  VectorXd z_pred = Zsig * weights_;
  z_pred(1) = atan2(sin(z_pred(1)), cos(z_pred(1)));

  // Calculate Measurement Covariance Matrix S and Cross Correlation Matrix Tc
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i = 0; i < Zsig.cols(); ++i) {
      VectorXd dZ = Zsig.col(i) - z_pred;
      dZ(1) = atan2(sin(dZ(1)), cos(dZ(1)));
      S += weights_(i) * dZ * dZ.transpose();

      VectorXd dX = Xsig_pred_.col(i) - x_;
      dX(3) = atan2(sin(dX(3)), cos(dX(3)));
      Tc += weights_(i) * dX * (dZ).transpose();
  }
  S(0, 0) += std_radr_ * std_radr_;
  S(1, 1) += std_radphi_ * std_radphi_;
  S(2, 2) += std_radrd_ * std_radrd_;

  // Calculate Kalman Gain K;
  MatrixXd S_inv = S.inverse();
  MatrixXd K = Tc * S_inv;

  // Update state mean and covariance matrix
  VectorXd y = meas_package.raw_measurements_ - z_pred;
  y(1) = atan2(sin(y(1)), cos(y(1)));
  x_ += K * y;
  x_(3) = atan2(sin(x_(3)), cos(x_(3)));
  P_ -= K * S * K.transpose();

  // Calculate the NIS
  NIS_radar_ = y.transpose() * S_inv * y;

  cout << "Zsig:" << endl << Zsig <<endl
       << "z_pred:" << endl << z_pred << endl
       << "S:" << endl << S << endl
       << "Tc:" << endl << Tc  << endl
       << "S_inv:" << endl << S_inv << endl
       << "K:" << endl << K << endl
       << "z:" << endl << meas_package.raw_measurements_ << endl
       << "y:" << endl << y << endl
       << "x:" << endl << x_ << endl
       << "P:" <<endl << P_ << endl;

  assert(x_.allFinite());
  assert(P_.allFinite());
  assert(P_.llt().info() != Eigen::NumericalIssue);
}
