#include "enemy_predictor/enemy_filter.h"
#include "rclcpp/rclcpp.hpp"

EnemyCKF::CKFConfig EnemyCKF::config_;
EnemyCKF::EnemyCKF() : sample_num_(2 * STATE_NUM), is_initialized_(false) {
   
    samples_.resize(sample_num_);
    sample_X.resize(sample_num_);
    sample_Z.resize(sample_num_);
    weights_.resize(sample_num_);
    
    double weight = 1.0 / (2 * STATE_NUM);
    for (int i = 0; i < sample_num_; ++i) {
        weights_[i] = weight;
    }
    last_timestamp_tmp = 0.0;
    last_timestamp_ = 0.0;
    Xe = Vx::Zero();
    Pe = Mxx::Identity();
    Pp = Mxx::Identity();
    Q = Mxx::Identity() * 0.1;
    R = Mzz::Identity() * 0.01;
    
}
void EnemyCKF::reset(const Eigen::Vector3d& position, double yaw, int _phase_id, 
                    double _timestamp) {
    angle_dis_ = M_PI / 2;
    
    // 直接初始化状态向量
    Xe = Vx::Zero();
    Xe[0] = position.x() + initial_radius[_phase_id % 2] * cos(yaw); // x
    Xe[1] = 0.0;  // vx
    Xe[2] = position.y() - initial_radius[_phase_id % 2] * sin(yaw); // y
    Xe[3] = 0.0;  // vy
    Xe[4] = yaw;  // yaw
    Xe[5] = 0.0;  // omega
    Xe[6] = initial_radius[0];  // r1
    Xe[7] = initial_radius[1];  // r2
    
    Pe = config_.config_Pe;
    Pp = Mxx::Identity();

    yaw_round_ = 0;
    last_yaw = yaw;

    last_timestamp_tmp = _timestamp;
    last_timestamp_ = _timestamp;
    is_initialized_ = true;
}

void EnemyCKF::update(const Eigen::Vector3d& position, double yaw, 
            double _timestamp, int _phase_id) {
    
    double dT = _timestamp - last_timestamp_;
    if (dT < 1e-6){
        dT = _timestamp - last_timestamp_tmp; // 说明是同一帧的两个armor
        RCLCPP_INFO(rclcpp::get_logger("EnemyCKF"), "The second armor in the same Frame, update CKF again!", dT);
    }
    RCLCPP_INFO(rclcpp::get_logger("EnemyCKF"), "update : timestamp - last_timestamp = %lf", dT);
    if(dT < 1e-6 || dT > 1.0){
        dT = 0.015;
        RCLCPP_INFO(rclcpp::get_logger("EnemyCKF"), "Invalid dT !!! Using default : %lf", dT);
    }
    Vz z;
    z << position.x(), position.y(), yaw;

    predict(dT);

    measure(z, _phase_id);

    correct(z);
    
    last_timestamp_tmp = last_timestamp_;
    last_timestamp_ = _timestamp;
}
// 预测特定!!装甲板位置
Eigen::Vector3d EnemyCKF::predictArmorPosition(double z, int phase_id, double dt) {   //先保留 timestamp 的接口

    double current_radius = Xe[6 + (phase_id % 2)];
    //double current_radius = 0.25;
    //double armor_yaw = Xe[4] + Xe[5]*dt + phase_id * angle_dis_;
    double armor_yaw = normalizeAngle(Xe[4] + Xe[5]*dt + phase_id * angle_dis_);
    
    double pred_x = Xe[0] + Xe[1]*dt - current_radius* cos(armor_yaw);
    double pred_y = Xe[2] + Xe[3]*dt + current_radius* sin(armor_yaw);
    double pred_z = z;
    
    return Eigen::Vector3d(pred_x, pred_y, pred_z);
}
Eigen::Vector3d EnemyCKF::predictCenterPosition(double z, double dt) {

    Vz res;
    res[0] = Xe[0] + Xe[1] * dt;  // x
    res[1] = Xe[2] + Xe[3] * dt;  // y
    res[2] = Xe[4] + Xe[5] * dt;  // yaw
    double pred_z = z;
    
    return Eigen::Vector3d(res[0], res[1], pred_z);
}
void EnemyCKF::initializeCKF() {
    if (!is_initialized_){
       samples_.resize(sample_num_);
       sample_X.resize(sample_num_);
       sample_Z.resize(sample_num_);
       weights_.resize(sample_num_);
       
       double weight = 1.0 / (2 * STATE_NUM);
       for (int i = 0; i < sample_num_; ++i) {
           weights_[i] = weight;
       }
       Pe = config_.config_Pe;
       Pp = Mxx::Identity();
       Xe = Vx::Zero();
       Q = Mxx::Identity() * 0.1;
       R = Mzz::Identity() * 0.01;
       is_initialized_ = true;
    }
}
// CKF核心算法
void EnemyCKF::SRCR_sampling(const Vx& x, const Mxx& P) {
    double sqrtn = sqrt(STATE_NUM);

    Eigen::LLT<Mxx> cholesky(P);
    Mxx S = cholesky.matrixL();
    
    for (int i = 0; i < STATE_NUM; ++i) {
        samples_[i] = x + sqrtn * S.col(i);
        samples_[i + STATE_NUM] = x - sqrtn * S.col(i);
    }
}
void EnemyCKF::predict(double dt) {
    if (!is_initialized_) {
        std::cerr << "CKF not initialized!" << std::endl;
        return;
    }
    
    //double dt = timestamp - last_timestamp_;

    if (dt <= 0) dt = 0.01;

    calcQ(dt);
    SRCR_sampling(Xe, Pe);

    Xp = Vx::Zero();
    
    for (int i = 0; i < sample_num_; ++i) {
        sample_X[i] = f(samples_[i], dt);

        Xp += weights_[i] * sample_X[i];
    }

    Pp = Mxx::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        Pp += weights_[i] * (sample_X[i] - Xp) * (sample_X[i] - Xp).transpose();
    }
    
    Pp += Q;
}
void EnemyCKF::measure(const Vz& z, int phase_id) {

    SRCR_sampling(Xp, Pp);

    calcR(z);
    
    Zp = Vz::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        sample_Z[i] = h(samples_[i], phase_id);
        Zp += weights_[i] * sample_Z[i];
    }
    Pzz = Mzz::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        Pzz += weights_[i] * (sample_Z[i] - Zp) * (sample_Z[i] - Zp).transpose();
    }
    Pzz += R;

}
void EnemyCKF::correct(const Vz& z) {
    Pxz = Mxz::Zero();
    for (int i = 0; i < sample_num_; ++i) {
        Pxz += weights_[i] * (sample_X[i] - Xp) * (sample_Z[i] - Zp).transpose();
    }
    
    K = Pxz * Pzz.inverse();

    /*// 计算残差，特别注意角度残差
    Vz residual = z - Zp;
    
    // 关键：角度残差需要特殊处理！
    // 不能简单相减，要考虑角度过0点的情况
    double angle_residual = z[2] - Zp[2];
    
    // 归一化角度残差到[-π, π]
    while (angle_residual > M_PI) angle_residual -= 2 * M_PI;
    while (angle_residual < -M_PI) angle_residual += 2 * M_PI;
    
    // 如果残差太大，可能是角速度预测不准，需要限制
    const double MAX_ANGLE_RESIDUAL = M_PI / 2;  // 最大允许90度残差
    if (fabs(angle_residual) > MAX_ANGLE_RESIDUAL) {
        RCLCPP_WARN(rclcpp::get_logger("EnemyCKF"), 
                   "Large angle residual: %lf rad, limiting to %lf", 
                   angle_residual, MAX_ANGLE_RESIDUAL);
        angle_residual = std::copysign(MAX_ANGLE_RESIDUAL, angle_residual);
    }
    
    residual[2] = angle_residual;*/

    Xe = Xp + K * (z - Zp);

    // Xe[4] = normalizeAngle(Xe[4]);
    
    // 角速度约束
    //const double MAX_OMEGA = M_PI * 3.0;  // 最大3 rad/s
    //Xe[5] = std::clamp(Xe[5], -MAX_OMEGA, MAX_OMEGA);

    Pe = Pp - K * Pzz * K.transpose();
}
// 系统模型
EnemyCKF::Vx EnemyCKF::f(const Vx& x, double dt) const {
    Vx ans = x;
    //double dt = timestamp - last_timestamp_;

    ans[0] += x[1] * dt;  // x += vx*dt
    ans[2] += x[3] * dt;  // y += vy*dt
    ans[4] += x[5] * dt;  // yaw += w*dt

    //ans[4] = normalizeAngle(ans[4]);

    return ans;
}
EnemyCKF::Vz EnemyCKF::h(const Vx& x, int phase_id) const {
    
    Vz result;
    double current_radius = x[6 + (phase_id % 2)];

    //double current_radius = 0.25;

    result[0] = x[0] - current_radius* cos(x[4] + phase_id * angle_dis_);  // x
    result[1] = x[2] + current_radius* sin(x[4] + phase_id * angle_dis_);  // y
    result[2] = x[4] + phase_id * angle_dis_;  // yaw

    //result[2] = normalizeAngle(result[2]);
    return result;
}
void EnemyCKF::calcQ(double dt) {
    static double dTs[4];
    dTs[0] = dt;
    for (int i = 1; i < 4; ++i) {
        dTs[i] = dTs[i - 1] * dt;
    }
    
    double q_x_x = dTs[3] / 3 * config_.Q2_X;      // dt³/3
    double q_x_vx = dTs[2] / 2 * config_.Q2_X;     // dt²/2
    double q_vx_vx = dTs[1] * config_.Q2_X;        // dt
    
    double q_y_y = dTs[3] / 3 * config_.Q2_Y;      // 区分X和Y方向
    double q_y_vy = dTs[2] / 2 * config_.Q2_Y;
    double q_vy_vy = dTs[1] * config_.Q2_Y;
    
    // 角度使用不同的噪声强度
    double q_yaw_yaw = dTs[3] / 3 * config_.Q2_YAW;
    double q_yaw_omega = dTs[2] / 2 * config_.Q2_YAW;
    double q_omega_omega = dTs[1] * config_.Q2_YAW;

    double radius_cov_avg = (Pe(6, 6) + Pe(7, 7)) / 2.0;
    double cov_factor = std::min(1.0, 10.0 * radius_cov_avg);  // 协方差大时保持较大Q
    double q_radius = config_.Q_r * dt * cov_factor;
    //设置最小值和最大值
    q_radius = std::max(q_radius, 1e-8 * dt);  // 最小值
    q_radius = std::min(q_radius, config_.Q_r * dt * 5.0);  // 最大值（5倍基础值）
    
    Q = Mxx::Zero();
    Q.block(0, 0, 2, 2) << q_x_x, q_x_vx, q_x_vx, q_vx_vx;
    Q.block(2, 2, 2, 2) << q_y_y, q_y_vy, q_y_vy, q_vy_vy;
    Q.block(4, 4, 2, 2) << q_yaw_yaw, q_yaw_omega, q_yaw_omega, q_omega_omega;

    Q(6, 6) = q_radius;
    Q(7, 7) = q_radius;
}
void EnemyCKF::calcR(const Vz& z) {
    Vz R_vec;
    R_vec << abs(config_.R_XYZ * z[0]), abs(config_.R_XYZ * z[1]),config_.R_YAW;
    R = R_vec.asDiagonal();
}
double EnemyCKF:: get_average(const std::vector<double>& vec) const {
    return std::accumulate(vec.begin(), vec.end(), 0.0) / vec.size();
}
