#include "enemy_predictor/enemy_predictor_node.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/opencv.hpp>

EnemyPredictor::EnemyPredictor(rclcpp::Node* node) 
    : node_(node)
{
    initBallistic();
    initFilterParams();
    initCommandParams();
    tf2_buffer = std::make_shared<tf2_ros::Buffer>(node_->get_clock());
    tf2_listener = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer);

    detector_sub = node_->create_subscription<vision_msgs::msg::Detection2DArray>(
        "/vision/tracked", rclcpp::QoS(10), 
        std::bind(&EnemyPredictor::detection_callback, this, std::placeholders::_1));
    
    control_pub = node_->create_publisher<rm_msgs::msg::Control>("enemy_predictor", rclcpp::SensorDataQoS());

    enemy_markers_pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
        "enemy_center_markers", 10);
    
    // "enemy_predictor" -> "basic_predictor"
    std::string shared_dir = ament_index_cpp::get_package_share_directory("basic_predictor");

    RCLCPP_INFO(node_->get_logger(), "shared_dir: %s", shared_dir.c_str());
    
    for(int i = 0; i < MAX_ENEMIES; i++){
        enemies_[i].class_id = i+1;
    }
}

void EnemyPredictor::initBallistic() {
   
    node_->declare_parameter<double>("ballistic.velocity", 28.0);
    node_->declare_parameter<double>("ballistic.drag_k", 0.01);
    
    double velocity = node_->get_parameter("ballistic.velocity").get_value<double>();
    double drag_k = node_->get_parameter("ballistic.drag_k").get_value<double>();
 
    ballistic_solver_ = std::shared_ptr<BallisticSolver>();

    ballistic_solver_->build(velocity, drag_k);
    
    if (ballistic_solver_->is_built()) {
        RCLCPP_INFO(node_->get_logger(), "弹道表构建成功");
    } else {
        RCLCPP_WARN(node_->get_logger(), "弹道表构建失败");
    }
}

void EnemyPredictor::initFilterParams() {

    auto declare_get = [this](const std::string& name, auto default_val) -> auto {
        using T = decltype(default_val);
        node_->declare_parameter<T>(name, default_val);
        return node_->get_parameter(name).template get_value<T>();
    };
    auto p_diag = declare_get("armor_ekf.P", std::vector<double>{0.025, 0.049, 0.001, 0.01, 0.01, 0.01});
    ArmorXYYAWEKF::config_.config_P = Eigen::Map<Eigen::VectorXd>(p_diag.data(), 6);
    
    auto q_diag = declare_get("armor_ekf.Q", std::vector<double>{0.025, 0.049, 0.001, 0.01, 0.01, 0.01});
    ArmorXYYAWEKF::config_.config_Q = Eigen::Map<Eigen::VectorXd>(q_diag.data(), 6);
    
    auto r_diag = declare_get("armor_ekf.R", std::vector<double>{0.00025, 0.00016, 0.0025});
    ArmorXYYAWEKF::config_.config_R = Eigen::Map<Eigen::VectorXd>(r_diag.data(), 3);
    // ZEKF 参数
    auto q_z = declare_get("armor_z_ekf.Q", std::vector<double>{0.01, 0.01});
    ZEKF::config_.config_Q = Eigen::Map<Eigen::VectorXd>(q_z.data(), 2);
    
    // 单个 double 参数
    auto r_z = declare_get("armor_z_ekf.R", 0.01);
    ZEKF::config_.config_R = Eigen::VectorXd(1);
    ZEKF::config_.config_R(0) = r_z;
    // Enemy CKF 协方差矩阵（8x8）
    auto pe = declare_get("enemy_ckf.Pe", std::vector<double>{
        0.25, 0.02, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.02, 0.64, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.25, 0.02, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.02, 0.64, 0.00, 0.00, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.09, 0.03, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.03, 0.25, 0.00, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04, 0.00,
        0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.04
    });
    EnemyCKF::config_.config_Pe = Eigen::Map<Eigen::MatrixXd>(pe.data(), 8, 8);

    EnemyCKF::config_.Q2_X = declare_get("enemy_ckf.Q2_X", 84.0);
    EnemyCKF::config_.Q2_Y = declare_get("enemy_ckf.Q2_Y", 84.0);
    EnemyCKF::config_.Q2_YAW = declare_get("enemy_ckf.Q2_YAW", 84.0);
    EnemyCKF::config_.R_XYZ = declare_get("enemy_ckf.R_XYZ", 0.01);
    EnemyCKF::config_.R_YAW = declare_get("enemy_ckf.R_YAW", 0.01);
    EnemyCKF::config_.Q_r = declare_get("enemy_ckf.Q_r", 0.01);
}
void EnemyPredictor::initCommandParams() {
    // 命令控制参数
    auto declare_get = [this](const std::string& name, auto default_val) -> auto {
        using T = decltype(default_val);
        node_->declare_parameter<T>(name, default_val);
        return node_->get_parameter(name).template get_value<T>();
    };
    cmd.high_spd_rotate_thresh = declare_get("high_spd_rotate_thresh", 0.30);
    cmd.rotate_thresh = declare_get("rotate_thresh", 0.25);
    cmd.yaw_thresh = declare_get("yaw_thresh", 0.01);
    params_.response_delay = declare_get("response_delay", 0.30);
    params_.shoot_delay = declare_get("shoot_delay", 0.30);
    auto yaw_offsets = declare_get("ballistic.stored_yaw_offset", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});
    cmd.stored_yaw_offsets = yaw_offsets;
    auto pitch_offsets = declare_get("ballistic.stored_pitch_offset", std::vector<double>{0.0, 0.0, 0.0, 0.0, 0.0});
    cmd.stored_pitch_offsets = pitch_offsets;
}

void EnemyPredictor::detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr detection_msg){

    std::vector<Detection, Eigen::aligned_allocator<Detection>> current_detections_{};
    std::vector<int>active_enemies_idx{};
    std::vector<int>active_armor_idx{};
    std::vector<cv::Point3f> object_points;
    time_det = node_-> now();
    double timestamp = time_det.seconds();
    const auto& detections = detection_msg->detections;
    time_image = detection_msg-> header.stamp;

    tf_.camara_to_base_link = getTrans("camera_optical_frame", "base_link", time_image);
    tf_.base_link_to_gimbal =  getTrans("base_link", "gimbal", time_image);
    
    cmd.cmd_mode = -1;
    cmd.booster_enable = 0;
    for(Enemy& enemy : enemies_){
        enemy.is_active = false;
    }
    for (const auto& detection : detections) {
        bool valid_det = true;
        if(detection.bbox.size_x / detection.bbox.size_y < 0.5 || detection.bbox.size_x / detection.bbox.size_y > 4){
            continue;
        }
        Detection det{};
        det.armor_idx = std::stoi(detection.id);
        det.area_2d = detection.bbox.size_x * detection.bbox.size_y;
        const auto& results_ = detection.results;
       
        for(const auto& res : results_){
            const auto& pos = res.pose;
            
            det.position = Eigen::Vector3d(pos.pose.position.x, pos.pose.position.y, pos.pose.position.z) ;
            det.orientation = Eigen::Quaterniond(pos.pose.orientation.w, pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z);
            //double yaw_cam = getYawfromQuaternion(pos.pose.orientation.w, pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z);
            det.yaw = getYawfromQuaternion(pos.pose.orientation.w, pos.pose.orientation.x, pos.pose.orientation.y, pos.pose.orientation.z);
            if(abs(det.yaw) > 1.05){
                valid_det = false;
            }
            //rm_msgs::msg::Imu imu_data = predictor_node->get_imu(time_image);
           
            //det.yaw = yaw_cam - imu_data.yaw;
           
            det.armor_class_id = std::stoi(res.hypothesis.class_id);
            
            if(det.armor_class_id % 10 == 1){
                object_points = large_object_points;
            }else{
                object_points = small_object_points;
            }
        }
        if(valid_det){
            updateArmorDetection(object_points, det, time_image);
            current_detections_.emplace_back(det);
        }
    }
    if(current_detections_.empty()){
        return;
    }
    ToupdateArmors(current_detections_, timestamp, active_armor_idx);
   
    EnemyManage(timestamp, time_image, active_enemies_idx, active_armor_idx);

    rm_msgs::msg::Control control_msg{};
    //如果真的出现没有目标的特殊情况，cmd不会更新，保持上一帧的控制指令，云台会停在原来的位置不动
    if (cmd.cmd_mode == 0 || cmd.cmd_mode == 1 || cmd.cmd_mode == 2){
        // 都开自瞄了肯定要控云台，只是发不发弹的区别
        //control_msg.flag = 1;
        //control_msg.vision_follow_id = enemies_[cmd.target_enemy_idx].class_id;
        //control_msg.pitch = cmd.cmd_pitch;
        //control_msg.yaw = cmd.cmd_yaw;
        //control_msg.one_shot_num = cmd.one_shot_num;
        //control_msg.rate = cmd.rate;
        rm_msgs::msg::Imu imu_data = predictor_node->get_imu(time_image);
        control_msg.imu.set__roll(imu_data.roll).set__pitch(cmd.cmd_pitch).set__yaw(cmd.cmd_yaw);
        control_msg.control_mode = 2;

        control_msg.booster_enable = cmd.booster_enable;
        //在这套架构下子节点不直接publish
        //control_pub-> publish(std::move(control_msg));
    }else{
        // MODE不是0/1/2，不控云台，虽然我现在不觉得会有这个else的情况
        control_msg.control_mode = 0;
        //control_pub-> publish(std::move(control_msg));
    }
    // rviz2可视化
    if (!enemy_markers_.markers.empty()) {
        enemy_markers_pub_->publish(enemy_markers_);
    }
}
