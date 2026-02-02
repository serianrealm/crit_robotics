#ifndef _ENEMY_CKF_H
#define _ENEMY_CKF_H
#include <iostream>
#include <Eigen/Dense>
#include <numeric> 


class EnemyCKF {
public:
    static const int STATE_NUM = 8;  // [x, vx, y, vy, yaw, vyaw, r1, r2]
    static const int OBSERVE_NUM = 3; // [x, y, yaw]
    
    using Vx = Eigen::Vector<double, STATE_NUM>;
    using Vz = Eigen::Vector<double, OBSERVE_NUM>;
    using Mxx = Eigen::Matrix<double, STATE_NUM, STATE_NUM>;
    using Mzz = Eigen::Matrix<double, OBSERVE_NUM, OBSERVE_NUM>;
    using Mxz = Eigen::Matrix<double, STATE_NUM, OBSERVE_NUM>;

    // 状态变量
    Vx Xe;  // 当前状态估计 [x, vx, y, vy, yaw, vyaw, r1, r2]
    Vx Xp;  // 预测状态
    
    // CKF变量
    Mxx Pe, Pp, Q;
    Mzz R, Pzz;
    Mxz Pxz, K;
    Vz Zp;
    
    int sample_num_ = 2 * STATE_NUM;
    std::vector<double> weights_;
    std::vector<Vx> samples_;
    std::vector<Vx> sample_X;
    std::vector<Vz> sample_Z;
    std::vector<double> initial_radius{0.25, 0.25};
    //double radius;
    
    double last_timestamp_tmp; //暂存一下last_timestamp_，同一帧 > 1个armor时可以都用来更新ckf
    double last_timestamp_;
    bool is_initialized_;

    double last_enemy_yaw_;  // 上一次的enemy_yaw值，用于检测跳变
    int yaw_unwrap_count_;   // yaw展开计数
    std::vector<int> phase_id_cnt = {0,0,0,0};

    // 机器人特定参数
    double angle_dis_;

    // 配置结构体
    struct CKFConfig {
        double Q2_X = 0.1;
        double Q2_Y = 0.1;
        double Q2_YAW = 0.01;
        double Q_r = 0.01;
        double R_XYZ = 0.01;
        double R_YAW = 0.001;
        Mxx config_Pe;
    };
    
    static CKFConfig config_;

    // 构造函数
    EnemyCKF();

    // 重置滤波器
    void reset(const Eigen::Vector3d& position, double yaw, int _phase_id, 
               double _timestamp);
    
    // 更新函数
    void update(const Eigen::Vector3d& position, double yaw, 
                double _timestamp, int _phase_id);
    
    // 预测特定装甲板位置
    Eigen::Vector3d predictArmorPosition(double z, int phase_id, double dt);
    Eigen::Vector3d predictCenterPosition(double z, double dt);
    // 初始化CKF
    void initializeCKF();
    
    // CKF核心算法
    void SRCR_sampling(const Vx& x, const Mxx& P);
    void predict(double dt);
    void measure(const Vz& z, int phase_id);
    void correct(const Vz& z);
    
    // 系统模型和观测模型
    Vx f(const Vx& x, double dt) const;
    Vz h(const Vx& x, int phase_id) const;
    
    // 噪声计算
    void calcQ(double dt);
    void calcR(const Vz& z);

    // 将角度归一化到[-π, π]
    double normalizeAngle(double angle) const {
        while (angle > M_PI) angle -= 2 * M_PI;
        while (angle < -M_PI) angle += 2 * M_PI;
        return angle;
    }
    
    // 获取归一化的yaw（用于外部接口）
    double getNormalizedYaw() const {
        return normalizeAngle(Xe[4]);
    }
    
    double wrapToReference(double angle, double reference) {
        // 计算差值并调整到[-π, π]
        double diff = angle - reference;
        
        // 将diff归一化到[-π, π]
        while (diff > M_PI) diff -= 2 * M_PI;
        while (diff < -M_PI) diff += 2 * M_PI;
        
        // 返回调整后的角度
        return reference + diff;
    }
    
    // 工具函数
    double get_average(const std::vector<double>& vec) const;
    bool is_initialized() const { return is_initialized_; }
};  
#endif