#ifndef UKF_H
#define UKF_H

#include "measurement_package.h"
#include "Eigen/Dense"
#include <vector>
#include <string>
#include <fstream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class UKF {
public:

  ///* initially set to false, set to true in first call of ProcessMeasurement
  bool is_initialized_;

  ///* if this is false, laser measurements will be ignored (except for init)
  const bool use_laser_;

  ///* if this is false, radar measurements will be ignored (except for init)
  const bool use_radar_;

  ///* State dimension
  const int n_x_;

  ///* Augmented state dimension
  const int n_aug_;

  ///* state vector: [pos1 pos2 vel_abs yaw_angle yaw_rate] in SI units and rad
  VectorXd x_;

  ///* state covariance matrix
  MatrixXd P_;

  ///* predicted sigma points matrix
  MatrixXd Xsig_pred_;

  ///* time when the state is true, in us
  long long time_us_;

  ///* Process noise standard deviation longitudinal acceleration in m/s^2
  const double std_a_;

  ///* Process noise standard deviation yaw acceleration in rad/s^2
  const double std_yawdd_;

  ///* Laser measurement noise standard deviation position1 in m
  const double std_laspx_;

  ///* Laser measurement noise standard deviation position2 in m
  const double std_laspy_;

  ///* Radar measurement noise standard deviation radius in m
  const double std_radr_;

  ///* Radar measurement noise standard deviation angle in rad
  const double std_radphi_;

  ///* Radar measurement noise standard deviation radius change in m/s
  const double std_radrd_ ;
  
  ///* Laser observation Matrix H for laser measurements
  MatrixXd H_laser_;
  
  ///* Laser measurement covariance matrix R
  MatrixXd R_laser_;

  ///* Weights of sigma points
  VectorXd weights_;

  ///* Sigma point spreading parameter
  const double lambda_;

  ///* the current NIS for radar
  double NIS_radar_;

  ///* the current NIS for laser
  double NIS_laser_;

  /**
   * Constructor
   */
  UKF();

  /**
   * Destructor
   */
  virtual ~UKF();

  /**
   * ProcessMeasurement
   * @param meas_package The latest measurement data of either radar or laser
   */
  void ProcessMeasurement(MeasurementPackage meas_package);
  
  /**
   * The linearized (Jacobian) H-inverse matrix for the radar, as a function
   * of the measuremento, used for the initialization of the P matrix
   * 
   * @param z_meaurement The measurement vector to convert to state space
   * @return The H-inverse matrix
   */
   MatrixXd HInvjRadar(const Eigen::VectorXd &z_measurement);

  /**
   * Prediction Predicts sigma points, the state, and the state covariance
   * matrix
   * @param delta_t Time between k and k+1 in s
   */
  void Predict(double delta_t);

  /**
   * Updates the state and the state covariance matrix using a laser measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateLidar(MeasurementPackage meas_package);

  /**
   * Updates the state and the state covariance matrix using a radar measurement
   * @param meas_package The measurement at k+1
   */
  void UpdateRadar(MeasurementPackage meas_package);
};

#endif /* UKF_H */
