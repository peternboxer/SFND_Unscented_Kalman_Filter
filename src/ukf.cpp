#include "ukf.h"
#include "Eigen/Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() 
{
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.6;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.5;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */

  time_us_ = 0;
  is_initialized_ = false;

  n_x_ = 5;
  n_aug_ = 7;
  Q_ = Eigen::MatrixXd(2,2);
  Xsig_aug_ = Eigen::MatrixXd(n_aug_, 2*n_aug_ + 1);
  
  lambda_ = 3 - n_aug_;

  Xsig_pred_ = Eigen::MatrixXd(n_x_, 2*n_aug_ + 1);
  Xsig_pred_.fill(0.0);
  
  weights_ = Eigen::VectorXd(2*n_aug_ +1);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
    // use first measurement to initialize predition
    if(!is_initialized_)
    {
      is_initialized_ = true;   // set to true to skip in the next iteration
      std::ofstream ofs("NIS_radar.txt");
      // initialize state covariance matrix with I
      P_ = Eigen::Matrix<double,5,5>::Identity();

      // LiDAR: x, y
      if(meas_package.sensor_type_ == MeasurementPackage::LASER)
      {
        double x = meas_package.raw_measurements_(0);
        double y = meas_package.raw_measurements_(1);
        x_ << x,y,0,0,0;

        P_(0,0) = std::pow(std_laspx_,2);
        P_(1,1) = std::pow(std_laspy_,2);
      }
      // Radar: rho, phi, rho_dot
      else
      {  
        double rho = meas_package.raw_measurements_(0);
        double phi = meas_package.raw_measurements_(1);
        double rho_dot = meas_package.raw_measurements_(2);   // host vehicle perspective

        // convert to px, py, v, yaw, yawd
        double x = rho*sin(phi);
        double y = rho*cos(phi);
        x_ << x, y, 1.25*rho_dot, phi,0;     // assume 1.25 times of the radar speed 
      }
      time_us_ = meas_package.timestamp_;
      return;
    }
    // convert to seconds
    double delta_t = (meas_package.timestamp_ - time_us_)/1e6;

    // update time 
    time_us_ = meas_package.timestamp_;
    Prediction(delta_t);

    if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    { UpdateLidar(meas_package);}
    else
    { UpdateRadar(meas_package);}
}

void UKF::Prediction(const double delta_t) 
{
  x_aug = VectorXd(n_aug_);
  P_aug = MatrixXd(n_aug_, n_aug_);

  Xsig_aug_ = MatrixXd(n_aug_, 2 * n_aug_ + 1);

  // augmented state vector
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  // augmented state covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;

  // noise covariance matrix
  Q_.fill(0.0);
  Q_(0,0) = std::pow(std_a_,2);
  Q_(1,1) = std::pow(std_yawdd_,2);
  P_aug.bottomRightCorner(2,2) = Q_;

  // cholesky decomposition
  MatrixXd A = P_aug.llt().matrixL();

  // augmented sigma points
  Xsig_aug_.fill(0.0);
  Xsig_aug_.col(0)  = x_aug;
  for (int i = 1; i< n_aug_ + 1; ++i) {
    Xsig_aug_.col(i)       = x_aug + sqrt(lambda_+n_aug_) * A.col(i-1);
    Xsig_aug_.col(i+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i-1);
  }

  // predict sigma points at k+1
  for (int i = 0; i< 2*n_aug_+1; ++i) {

    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double a = Xsig_aug_(5,i);
    double yawa = Xsig_aug_(6,i);

    double px_p, py_p;

    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    } else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    px_p = px_p + 0.5*a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*a*delta_t*delta_t * sin(yaw);
    v_p = v_p + a*delta_t;

    yaw_p = yaw_p + 0.5*yawa*delta_t*delta_t;
    yawd_p = yawd_p + yawa*delta_t;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }
  
  // determine weights
  weights_ = Eigen::VectorXd(2*n_aug_+1);
  weights_(0) = lambda_ / (lambda_ + n_aug_);
  for (size_t i=1; i<2*n_aug_+1; ++i) {  // 2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  // mean state vector
  x_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points
    x_ += weights_(i) * Xsig_pred_.col(i);
  }

  // mean state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < 2 * n_aug_ + 1; ++i) {  // iterate over sigma points

    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    while (x_diff(3)> M_PI)
    {
      x_diff(3)-=2.*M_PI;
    } 
    while (x_diff(3)<-M_PI)
    {
      x_diff(3)+=2.*M_PI;
    } 
    P_ += weights_(i) * x_diff * x_diff.transpose() ;
  }
}

void UKF::UpdateLidar(MeasurementPackage meas_package) 
{
  int n_z = 2;    // x, y
  Eigen::MatrixXd Z_sig = Eigen::MatrixXd(n_z, 2*n_aug_+1);    // 2x15
  Z_sig.fill(0.0);
  Eigen::VectorXd z_pred(n_z);   // 2x1
  z_pred.fill(0.0);
  
  for(size_t i = 0; i<2*n_aug_+1; ++i)
  {
    Z_sig(0,i) = Xsig_pred_(0,i);
    Z_sig(1,i) = Xsig_pred_(1,i);
    z_pred += weights_(i) * Z_sig.col(i);
  }

  // Predicted Covariance
  Eigen::Matrix2d S;
  S.fill(0.0);
  for(size_t i=0; i<2*n_aug_+1; ++i)
  {
    Eigen::VectorXd diff_z = Z_sig.col(i) - z_pred;
    S += weights_(i) * diff_z * diff_z.transpose();
  }

  // Sensor Covariance Matrix
  Eigen::Matrix2d R;
  R.fill(0.0);
  R(0,0) = std::pow(std_laspx_,2);
  R(1,1) = std::pow(std_laspy_,2);
  S += R;
  
  // Correlation Matrix
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(size_t i=0; i<2*n_aug_+1; ++i)
  {
    Eigen::VectorXd z_diff = Z_sig.col(i) - z_pred;
    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  Eigen::MatrixXd K = Tc * S.inverse();
  
  // new incoming data
  Eigen::Vector2d z_new = meas_package.raw_measurements_;

  // Update state
  x_ += K*(z_new - z_pred);    // 5x1
  P_ -= K*S*K.transpose();    // 5x5

  // NIS - DOF=2
  auto z_diff = z_new - z_pred;
  auto e = z_diff.transpose()*S.inverse()*z_diff;
  nis_lidar.emplace_back(e);
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {

  int n_z = 3;  // rho, phi, rho_dot
  Eigen::MatrixXd Z_sig = Eigen::MatrixXd(n_z, 2*n_aug_+1);
  Z_sig.fill(0.0);
  Eigen::VectorXd z_pred(n_z);

  for(size_t i=0; i<2*n_aug_+1; ++i)
  {
    double Px = Xsig_pred_(0,i);
    double Py = Xsig_pred_(1,i);
    double V = Xsig_pred_(2,i);
    double Yaw = Xsig_pred_(3,i);

    double v1 = cos(Yaw)*V;
    double v2 = sin(Yaw)*V;

    // transform into measurement space
    Z_sig(0,i) = std::sqrt(std::pow(Px,2) + std::pow(Py,2));
    Z_sig(1,i) = atan2(Py, Px);
    Z_sig(2,i) = (Px*v1 + Py*v2)/sqrt(std::pow(Px,2) + std::pow(Py,2));
  }

  // mean state vector
  z_pred.fill(0.0);
  for(size_t i=0; i<2*n_aug_+1; ++i)
  {
    z_pred += weights_(i)*Z_sig.col(i);
  }

  // mean Covariance matrix
  Eigen::Matrix3d S;
  S.fill(0.0);
  for(size_t i=0; i<2*n_aug_+1; ++i)
  {
    Eigen::VectorXd z_diff = Z_sig.col(i) - z_pred;
    while(z_diff(1) > M_PI)
    {
      z_diff(1) -= 2.*M_PI;
    }
    while(z_diff(1) < -M_PI)
    {
      z_diff(1) += 2.*M_PI;
    }
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // Sensor Covariance Matrix
  Eigen::Matrix3d R;
  R.fill(0.0);
  R(0,0) = std::pow(std_radr_,2);
  R(1,1) = std::pow(std_radphi_,2);
  R(2,2) = std::pow(std_radrd_,2);
  S  += R;

  // Update state
  // Correlation Matrix
  Eigen::MatrixXd Tc = Eigen::MatrixXd(n_x_,n_z);
  Tc.fill(0.0);
  for(size_t i=0; i < 2 * n_aug_+1; ++i)
  {
    Eigen::VectorXd z_diff = Z_sig.col(i) - z_pred;
    while(z_diff(1) > M_PI)
    {
      z_diff(1) -= 2.*M_PI;
    }
    while(z_diff(1) < -M_PI)
    {
      z_diff(1) += 2.*M_PI;
    }

    Eigen::VectorXd x_diff = Xsig_pred_.col(i) - x_;
    while(x_diff(3) > M_PI)
    {
      x_diff(3) -= 2*M_PI;
    }
    while(x_diff(3) < -M_PI)
    {
      x_diff(3) += 2*M_PI;
    }

    //std::cout<<x_diff<<"\n\n";
    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  Eigen::MatrixXd K = Tc * S.inverse();
  
  // new incoming measurements
  Eigen::Vector3d z_new = meas_package.raw_measurements_;

  Eigen::VectorXd z_diff = z_new - z_pred;
  // normalize angle
  while(z_diff(1) > M_PI)
  {
    z_diff(1) -= 2.*M_PI;
  }
  while(z_diff(1) < -M_PI)
  {
    z_diff(1) += 2.*M_PI;
  }

  // Update state
  x_ += K * z_diff;
  P_ -=  K*S*K.transpose();

  // NIS - DOF=3
  auto e = z_diff.transpose()*S.inverse()*z_diff;
  nis_radar.emplace_back(e);
}