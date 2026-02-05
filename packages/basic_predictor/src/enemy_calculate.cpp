#include <Eigen/Dense>
#include "enemy_predictor/enemy_predictor_node.h"



Eigen::Isometry3d EnemyPredictor::getTrans(const std::string& source_frame, const std::string& target_frame, 
                                               rclcpp::Time timestamp_image) {
    
    geometry_msgs::msg::TransformStamped t;
    try {
        t = tf2_buffer->lookupTransform(
            target_frame,
            source_frame,
            timestamp_image,
            rclcpp::Duration::from_seconds(0.5)
        );
    } catch (const std::exception& ex) {
        printf(
            "Could not transform %s to %s: %s",
            source_frame.c_str(),
            target_frame.c_str(),
            ex.what()
        );
    }
    // to Eigen::Isometry3d
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = Eigen::Vector3d(
        t.transform.translation.x,
        t.transform.translation.y,
        t.transform.translation.z
    );
    Eigen::Quaterniond quat(
        t.transform.rotation.w,
        t.transform.rotation.x,
        t.transform.rotation.y,
        t.transform.rotation.z
    );
    transform.rotate(quat);
    
    return transform;
    //return tf2::transformToEigen(t);
} 
void EnemyPredictor::updateArmorDetection(std::vector<cv::Point3f> object_points,
                                              Detection& det,
                                              rclcpp::Time timestamp_image) {
    //std::vector<cv::Point2f> reprojected_points;

    //cv::Mat tvec = (cv::Mat_<double>(3, 1) << 
    //                det.position.x(), 
    //                det.position.y(), 
    //                det.position.z());
    //cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
//
    //double roll = det.orientation(1);
    //double pitch = det.orientation(2);
    //double yaw = det.orientation(3);
    //RCLCPP_INFO(get_logger(), "roll = %lf", roll);
    //RCLCPP_INFO(get_logger(), "pitch = %lf", pitch);
    //RCLCPP_INFO(get_logger(), "yaw = %lf", yaw);

    //double cr = cos(roll), sr = sin(roll);
    //double cp = cos(pitch), sp = sin(pitch);
    //double cy = cos(yaw), sy = sin(yaw);
//
    //cv::Mat R = (cv::Mat_<double>(3, 3) <<
    //    cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr,
    //    sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr,
    //    -sp,    cp*sr,             cp*cr);
//
    //cv::Rodrigues(R, rvec);
//
    //cv::projectPoints(object_points, rvec, tvec, 
    //                   visualize_.camera_matrix, visualize_.dist_coeffs, reprojected_points);

    //const cv::Point2f& p0 = reprojected_points[0];
    //const cv::Point2f& p1 = reprojected_points[1];
    //const cv::Point2f& p2 = reprojected_points[2];
    //const cv::Point2f& p3 = reprojected_points[3];

    // 鞋带公式
    //double area = 0.0;
    //area += p0.x * p1.y - p1.x * p0.y;
    //area += p1.x * p2.y - p2.x * p1.y;
    //area += p2.x * p3.y - p3.x * p2.y;
    //area += p3.x * p0.y - p0.x * p3.y;
    //det.area_2d = std::abs(area) / 2.0;

    //Eigen::Vector3d camera_tvec_eigen = Eigen::Map<Eigen::Vector3d>(visualize_.camera_tvec.ptr<double>());
    //visualize_.camara_to_base_link = getTrans("camera_optical_frame", "base_link", timestamp_image);
    
    //det.position = visualize_.camara_to_base_link * det.position;  //camera to base_link
    //visualizeAimCenter(det.position, cv::Scalar(225, 0, 225));
    enemy_markers_.markers.clear();
    
    visualization_msgs::msg::Marker yaw_marker;
    yaw_marker.header.frame_id = "base_link";
    yaw_marker.header.stamp = node_->now();
    yaw_marker.ns = "tracker_yaw";
    yaw_marker.id = det.armor_class_id * 1000 + det.armor_class_id * 10 + 1; // 唯一ID
    
    yaw_marker.type = visualization_msgs::msg::Marker::ARROW;
    yaw_marker.action = visualization_msgs::msg::Marker::ADD;
    
    // 箭头的起始点（tracker当前位置）
    geometry_msgs::msg::Point start_point;
    start_point.x = det.position.x();
    start_point.y = det.position.y();
    start_point.z = det.position.z();
    
    // 箭头的结束点（根据yaw计算方向）
    geometry_msgs::msg::Point end_point;
    double arrow_length = 0.3; // 箭头长度
    double yaw_rad = det.yaw; // tracker的yaw（弧度）
    end_point.x = start_point.x - arrow_length * cos(yaw_rad);
    end_point.y = start_point.y + arrow_length * sin(yaw_rad);
    end_point.z = start_point.z; // 保持在相同高度
    
    yaw_marker.points.push_back(start_point);
    yaw_marker.points.push_back(end_point);
    
    yaw_marker.scale.x = 0.02;  // 箭头杆直径
    yaw_marker.scale.y = 0.04;  // 箭头头直径
    yaw_marker.scale.z = 0.1;   // 箭头头长度

     // 橙色表示朝向箭头
    yaw_marker.color.r = 1.0;
    yaw_marker.color.g = 0.5;
    yaw_marker.color.b = 0.0;
    yaw_marker.color.a = 0.9;
    
    yaw_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(yaw_marker);
    // -------------当前armor位置-----------------------------------
    visualization_msgs::msg::Marker tracker_marker;
    tracker_marker.header.frame_id = "base_link";
    tracker_marker.header.stamp = node_->now();
    tracker_marker.ns = "tracker_current";
    tracker_marker.id = det.armor_class_id * 1000 + det.armor_class_id * 10; // 唯一ID
    
    tracker_marker.type = visualization_msgs::msg::Marker::SPHERE;
    tracker_marker.action = visualization_msgs::msg::Marker::ADD;
    
    // 使用tracker当前位置
    tracker_marker.pose.position.x = det.position.x();
    tracker_marker.pose.position.y = det.position.y();
    tracker_marker.pose.position.z = det.position.z();
    tracker_marker.pose.orientation.w = 1.0;
    
    tracker_marker.scale.x = 0.06;  // 稍微小一点，与装甲板区分
    tracker_marker.scale.y = 0.06;
    tracker_marker.scale.z = 0.06;
    
    // 紫色表示tracker当前位置
    tracker_marker.color.r = 0.8;   // 紫色：红+蓝
    tracker_marker.color.g = 0.0;
    tracker_marker.color.b = 0.8;
    tracker_marker.color.a = 0.9;
    
    tracker_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(tracker_marker);
}
//--------------------------------Tracking with Armor Filter--------------------------------------------------
EnemyPredictor::ArmorTracker::ArmorTracker(int tracker_idx, 
                                              int armor_class_id,
                                              const Eigen::Vector3d& init_pos, 
                                              double timestamp,
                                              double armor_yaw,
                                              double area_2d)
    : tracker_idx(tracker_idx), 
      armor_class_id(armor_class_id), 
      yaw(armor_yaw), 
      last_yaw(armor_yaw),
      area_2d(area_2d) {
    
    phase_id = -1;
    position = init_pos;
    last_position = init_pos;
    predicted_position = init_pos;
    position_history.push_back(init_pos);
    last_update_time = timestamp;
    Eigen::Vector3d to_ekf_xyyaw = Eigen::Vector3d(init_pos.x(), init_pos.y(), armor_yaw);
    ekf.init(to_ekf_xyyaw, timestamp);
}


void EnemyPredictor::ArmorTracker::update(const Eigen::Vector3d& new_position, 
                                            int armor_class_id_, double timestamp,
                                            double armor_yaw) {
    last_position = position;
    position = new_position;
    armor_class_id = armor_class_id_;
    // 处理yaw过零（模仿上一版本）
    if (armor_yaw - last_yaw < -M_PI * 1.5) {
        yaw_round++;
    } else if (armor_yaw - last_yaw > M_PI * 1.5) {
        yaw_round--;
    }
    last_yaw = armor_yaw;
    yaw = armor_yaw + yaw_round * 2 * M_PI;
    
    // 更新EKF
    Eigen::Vector3d new_xyyaw = Eigen::Vector3d(new_position.x(), new_position.y(), armor_yaw);
    ekf.update(new_xyyaw, timestamp);
    ZEKF::Vz z_obs;
    z_obs(0) = new_position(2);
    zekf.update(z_obs, timestamp);
    is_active = true;
    // 更新历史
    position_history.push_back(new_position);
    if (position_history.size() > 100) {
        position_history.erase(position_history.begin());
    }
    
    last_update_time = timestamp;
    missing_frames = 0;
}
void EnemyPredictor::ToupdateArmors(const std::vector<Detection, Eigen::aligned_allocator<Detection>>& detections, double timestamp,
                                        std::vector<int>& active_armor_idx) {
    
    if (detections.empty()) {
        RCLCPP_INFO(get_logger(), "No Armor This Frame");
        return;
    }
  
    for(size_t i = 0; i < detections.size(); i++){

        enemies_[detections[i].armor_class_id % 10 - 1].is_active = true;
        bool has_history_tracker = false;

        for (int j = 0; j < armor_trackers_.size(); j++) {

            armor_trackers_[j].is_active = false;
            
            if(armor_trackers_[j].tracker_idx == detections[i].armor_idx){
                active_armor_idx.push_back(j);
               
                armor_trackers_[j].update(detections[i].position, detections[i].armor_class_id, timestamp, detections[i].yaw);
        
                has_history_tracker = true;
                armor_trackers_[j].is_active = true;
        
                break;
            }
        }   
        if(has_history_tracker == false){
            create_new_tracker(detections[i], timestamp, active_armor_idx);
        }
    }
}

void EnemyPredictor::updateEnemy(Enemy& enemy, double timestamp, std::vector<int>& active_armor_idx) {
    
    std::vector<ArmorTracker*> active_armors_this_enemy;

    for (int idx : active_armor_idx) {
        if(armor_trackers_[idx].armor_class_id % 10 == enemy.class_id){
            active_armors_this_enemy.push_back(&armor_trackers_[idx]);
        }
    }
    for(ArmorTracker* tracker : active_armors_this_enemy){
        findBestPhaseForEnemy(enemy, *tracker, active_armors_this_enemy);
        //RCLCPP_INFO(get_logger(), "Armor(%d), Phase_id = %d, area_2d = %lf", tracker->tracker_idx, tracker->phase_id, tracker->area_2d);
    }
    if(!enemy.enemy_ckf.is_initialized_){
        enemy.enemy_ckf.initializeCKF();
        enemy.enemy_ckf.reset(active_armors_this_enemy[0] -> position, active_armors_this_enemy[0] -> yaw, active_armors_this_enemy[0] -> phase_id, timestamp);
    }
    else if(enemy.missing_frame > 20 || enemy.is_valid == false){
        enemy.enemy_ckf.reset(active_armors_this_enemy[0] -> position, active_armors_this_enemy[0] -> yaw, active_armors_this_enemy[0] -> phase_id, timestamp);
    }
    else{
        double sum_z = 0.0;
        for(ArmorTracker* tracker : active_armors_this_enemy){
           enemy.enemy_ckf.update(tracker->position, tracker->yaw, timestamp, tracker->phase_id);
           sum_z += tracker -> position(2);
        }
        enemy.center(2) = sum_z / active_armors_this_enemy.size();
    }
    
}

void EnemyPredictor::EnemyManage(double timestamp, rclcpp::Time timestamp_image, 
                                     std::vector<int>& active_enemies_idx, std::vector<int>& active_armor_idx) {

    for(Enemy& enemy : enemies_){
        if(enemy.is_active){
            updateEnemy(enemy, timestamp, active_armor_idx);
            active_enemies_idx.push_back(enemy.class_id - 1); 
            enemy.missing_frame = 0;
        }else{
            enemy.missing_frame ++;
            if(enemy.missing_frame > 15){
                enemy.reset();
                enemy.is_valid = false;
            }
        }
    }
    int target_enemy_idx = -1;

    if(active_enemies_idx.size()== 0){
        return;
    }
    //是否需要考虑操作手按right键，但这一帧没有detect到操作手正在tracking的enemy？？？
    if(active_enemies_idx.size()== 1){
        target_enemy_idx = active_enemies_idx[0];
        //RCLCPP_INFO(get_logger(),"target_enemy_idx:%d",target_enemy_idx);
    }
    else if(active_enemies_idx.size() > 1){
        //基于到准星距离（操作手）的决策
        double enemy_to_heart_min = 10000.0;
        if(cmd.last_target_enemy_idx != -1 && params_.right_press == true){
            target_enemy_idx = cmd.last_target_enemy_idx;
        }
        else{
            for(int i = 0; i < active_enemies_idx.size(); i++){
                //如果某个enemy只存在过一个装甲板的detection,使用const_radius但只做ekf得到cmd结果
                //那么此时的enemy.center不准，但是choose enemy的时候需要使用enemy.center
                RCLCPP_INFO(get_logger(),"Start To Choose Target");
                Eigen::Vector3d enemy_center_cam = visualize_.camara_to_base_link.inverse() *enemies_[active_enemies_idx[i]].center;
               
                // DEBUG!!!!!!!!!!!! base_link to camera ,but distance???????? 
                std::vector<cv::Point2f> reprojected_points;
                std::vector<cv::Point3f> points_3d;
                cv::Point3f point_3d(enemy_center_cam.x(), enemy_center_cam.y(), enemy_center_cam.z());
                points_3d.push_back(point_3d);
                
                cv::projectPoints(points_3d, visualize_.camera_rvec, visualize_.camera_tvec, 
                          visualize_.camera_matrix, visualize_.dist_coeffs, reprojected_points);
                
                if(reprojected_points.size() > 0){
                    // Attention: camera和图传中心不一样，记得获取图传准星的标定结果！！！
                    float dx = static_cast<float>(reprojected_points[0].x - visualize_.camera_heart.x);
                    float dy = static_cast<float>(reprojected_points[0].y - visualize_.camera_heart.y);
                    double dis = std::sqrt(dx * dx + dy * dy);
                    if(dis < enemy_to_heart_min){
                        enemy_to_heart_min = dis;
                        target_enemy_idx = active_enemies_idx[i];

                    }
                }
                target_enemy_idx = active_enemies_idx[i];
            }
            RCLCPP_INFO(get_logger(),"Finish To Choose Target");
        }
    }
    getCommand(enemies_[target_enemy_idx], timestamp, timestamp_image, active_armor_idx);
}
    
void EnemyPredictor::findBestPhaseForEnemy(Enemy& enemy, ArmorTracker& tracker, std::vector<ArmorTracker*> active_armors_this_enemy) {

    if(!enemy.is_valid){
        tracker.phase_id = 0;
        enemy.is_valid = true; // 每个enemy有了第一个armor时，第一个armor phase_id = 0, enemy.is_valid = true
        RCLCPP_INFO(get_logger(), "This enemy (%d) has its first armor, Give Phase_id %d",enemy.class_id, tracker.phase_id);
        return;
    }
    bool by_relative = false;
    int candidate_phase = -1;
    for(size_t i = 0; i < active_armors_this_enemy.size(); i++){

        double angle_diff = normalize_angle(tracker.yaw - active_armors_this_enemy[i] -> yaw);
        
        if(active_armors_this_enemy[i]->phase_id != -1 && abs(angle_diff) > 1.4 && abs(angle_diff) < 1.75){
           if(angle_diff > 0){
              candidate_phase  = active_armors_this_enemy[i]->phase_id + 1;
              if(candidate_phase > 3){
                 candidate_phase = 0;
               }
            }
            else if(angle_diff < 0){
                candidate_phase = active_armors_this_enemy[i]->phase_id - 1;

                if(candidate_phase < 0){
                  candidate_phase = 3;
                }
            }
            by_relative = true;
        }
    }
    if(!by_relative){
        
        double angle_diff = normalize_angle(tracker.yaw - normalize_angle(enemy.enemy_ckf.Xe(4)));

        if(angle_diff >= -M_PI/4 && angle_diff < M_PI/4){
            candidate_phase = 0; // 前方
        }
        else if(angle_diff >= M_PI/4 && angle_diff < 3*M_PI/4){
            candidate_phase = 1; // 左侧
        }
        else if(angle_diff >= -3*M_PI/4 && angle_diff < -M_PI/4){
            candidate_phase = 3; // 右侧
        }
        else{
            candidate_phase = 2; // 后方
        }
        //RCLCPP_INFO(get_logger(), "Don't Have Useful relative armor!!!!!! candidate_phase = %d", candidate_phase);

    }
    if(tracker.phase_id != -1 && tracker.phase_id != candidate_phase){
        tracker.phase_id_cnt ++;
        if(tracker.phase_id_cnt > 4){
            tracker.phase_id = candidate_phase;
            tracker.phase_id_cnt = 0;
        }
    }
    else{
        tracker.phase_id = candidate_phase;
    }
    RCLCPP_INFO(get_logger(), "Give phase_id : %d", tracker.phase_id);
}
int EnemyPredictor::ChooseMode(Enemy &enemy, double timestamp){
    if(abs(enemy.enemy_ckf.Xe(5)) > cmd.high_spd_rotate_thresh){
       return 2;
    }
    else if(abs(enemy.enemy_ckf.Xe(5)) > cmd.rotate_thresh){
       return 1;
    }
    else{
        return 0;
    }
}
void EnemyPredictor::getCommand(Enemy& enemy, double timestamp, rclcpp::Time timestamp_image, std::vector<int>& active_armor_idx){
    
    cmd.cmd_mode = ChooseMode(enemy, timestamp);
    RCLCPP_INFO(get_logger(),"MODE :%d",cmd.cmd_mode);

    cmd.target_enemy_idx = enemy.class_id -1;
    cmd.last_target_enemy_idx = cmd.target_enemy_idx;

    auto predict_func_double = [this, &enemy](ArmorTracker& tracker, double time_offset, double timestamp) -> Eigen::Vector3d{
        return FilterManage(enemy, time_offset, tracker, timestamp);
    };

    auto predict_func_ckf = [this](Enemy& enemy_tmp, double time_offset, int phase_id) -> Eigen::Vector3d{
        return enemy_tmp.enemy_ckf.predictArmorPosition(enemy_tmp.center(2), phase_id, time_offset);
    };

    std::vector<ArmorTracker*> active_trackers;

    for(int idx : active_armor_idx){
        if(armor_trackers_[idx].armor_class_id % 10 == enemy.class_id){
            active_trackers.push_back(&armor_trackers_[idx]);
        }
    }
    //------------------- Visualize for Debug (rviz2) -------------------------
    //enemy_markers_.markers.clear();
    //for(ArmorTracker* tracker : active_trackers){
    //    visualization_msgs::msg::Marker yaw_marker;
    //    yaw_marker.header.frame_id = "base_link";
    //    yaw_marker.header.stamp = node_->now();
    //    yaw_marker.ns = "tracker_yaw";
    //    yaw_marker.id = enemy.class_id * 1000 + tracker->tracker_idx * 10 + 1; // 唯一ID
    //    
    //    yaw_marker.type = visualization_msgs::msg::Marker::ARROW;
    //    yaw_marker.action = visualization_msgs::msg::Marker::ADD;
    //    
    //    // 箭头的起始点（tracker当前位置）
    //    geometry_msgs::msg::Point start_point;
    //    start_point.x = tracker->position.x();
    //    start_point.y = tracker->position.y();
    //    start_point.z = tracker->position.z();
    //    
    //    // 箭头的结束点（根据yaw计算方向）
    //    geometry_msgs::msg::Point end_point;
    //    double arrow_length = 0.3; // 箭头长度
    //    double yaw_rad = tracker->yaw; // tracker的yaw（弧度）
    //    end_point.x = start_point.x - arrow_length * cos(yaw_rad);
    //    end_point.y = start_point.y + arrow_length * sin(yaw_rad);
    //    end_point.z = start_point.z; // 保持在相同高度
    //    
    //    yaw_marker.points.push_back(start_point);
    //    yaw_marker.points.push_back(end_point);
    //    
    //    yaw_marker.scale.x = 0.02;  // 箭头杆直径
    //    yaw_marker.scale.y = 0.04;  // 箭头头直径
    //    yaw_marker.scale.z = 0.1;   // 箭头头长度
    //
    //     // 橙色表示朝向箭头
    //    yaw_marker.color.r = 1.0;
    //    yaw_marker.color.g = 0.5;
    //    yaw_marker.color.b = 0.0;
    //    yaw_marker.color.a = 0.9;
    //    
    //    yaw_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    //    enemy_markers_.markers.push_back(yaw_marker);
    //}
    // --------------------------To show yaw in all mode -----------------------------------
    cv::putText(visualize_.armor_img, 
                 cv::format("yaw = %.2f", enemy.enemy_ckf.Xe(4)),  // 新增数据
                 cv::Point(50, 140),  // y坐标下移40像素
                 cv::FONT_HERSHEY_SIMPLEX,
                 1.0,
                 cv::Scalar(0, 255, 0),
                 2);
    //MODE = 2, 瞄中心，不跟随armor
    if(cmd.cmd_mode == 2){
        cv::putText(visualize_.armor_img, 
                        cv::format("W = %.2f", enemy.enemy_ckf.Xe(5)),  // 格式化文本
                        cv::Point(50, 100),              // 位置
                        cv::FONT_HERSHEY_SIMPLEX,        // 字体
                        1.0,                             // 大小
                        cv::Scalar(0, 0, 255),           // 颜色
                        2);                              // 粗细
        // 小陀螺瞄中心不需要last_armor_idx信息，进入MODE = 2时可以直接让last_armor_idx = -1
        cmd.last_armor_idx = -1;

        Eigen::Vector3d armor_center_pre = Eigen::Vector3d(0, 0, 0);
        Eigen::Vector3d armor_xyyaw_pre = Eigen::Vector3d(0, 0, 0);

        //这里只是算个t_fly,可以先随便算一个armor的，时间都差不多
        auto ball_res = calc_ballistic_second((params_.response_delay + params_.shoot_delay), timestamp_image, timestamp, 0, enemy, predict_func_ckf);
        Eigen::Vector3d enemy_center_pre = enemy.enemy_ckf.predictCenterPosition(enemy.center(2), ball_res.t + params_.response_delay + params_.shoot_delay);
        
        for(int i = 0; i < 4; i++){
            
           armor_center_pre = enemy.enemy_ckf.predictArmorPosition(enemy.center(2), i, (ball_res.t + params_.response_delay + params_.shoot_delay));

            //不管给不给发弹指令都跟随enemy中心
            double x = enemy_center_pre.x();
            double y = enemy_center_pre.y();
            double z = enemy_center_pre.z();
    
            auto ball_center = ballistic_solver_->query(x, y, z);
            
            cmd.cmd_pitch = ball_center.pitch;
            cmd.cmd_yaw = ball_center.yaw;

            double angle = angleBetweenVectors(enemy_center_pre, armor_center_pre);
            if (std::abs(angle) < cmd.yaw_thresh){
               
                cmd.booster_enable = 1;
                // ------------------可视化高速小陀螺时准备击打的预测点-------------------
                //visualizeAimCenter(armor_center_pre, cv::Scalar(0, 0, 255));
                geometry_msgs::msg::Pose pose_pre = vectorToPose(armor_center_pre);
                predictor_node->annotate(pose_pre);
                
                enemy_markers_.markers.clear();
                visualization_msgs::msg::Marker aim_marker;
                aim_marker.header.frame_id = "base_link";
                aim_marker.header.stamp = node_->now();
                aim_marker.ns = "tracker_current";
                aim_marker.id = enemy.class_id * 1000 + 50; // 唯一ID
                
                aim_marker.type = visualization_msgs::msg::Marker::SPHERE;
                aim_marker.action = visualization_msgs::msg::Marker::ADD;
                
                // 使用tracker当前位置
                aim_marker.pose.position.x = armor_center_pre.x();
                aim_marker.pose.position.y = armor_center_pre.y();
                aim_marker.pose.position.z = armor_center_pre.z();
                aim_marker.pose.orientation.w = 1.0;
                
                aim_marker.scale.x = 0.10;  // 稍微小一点，与装甲板区分
                aim_marker.scale.y = 0.10;
                aim_marker.scale.z = 0.10;
                
                // Red
                aim_marker.color.r = 1.0; 
                aim_marker.color.g = 0.0;
                aim_marker.color.b = 0.0;
                aim_marker.color.a = 0.9;
                
                aim_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
                enemy_markers_.markers.push_back(aim_marker);
                break; 
            }
            else{
                cmd.booster_enable = 0;
            }
        }
    }
    //MODE = 1/0，跟随armor
    else if(cmd.cmd_mode == 1 || cmd.cmd_mode == 0){
        cv::putText(visualize_.armor_img, 
                    cv::format("W = %.2f", enemy.enemy_ckf.Xe(5)),  // 格式化文本
                    cv::Point(50, 100),              // 位置
                    cv::FONT_HERSHEY_SIMPLEX,        // 字体
                    1.0,                             // 大小
                    cv::Scalar(0, 255, 255),         // 颜色
                    2);                              // 粗细
    
        if(active_trackers.size() == 0){
            cmd.last_armor_idx = -1;
            RCLCPP_WARN(this->get_logger(), "No active trackers found");
            return;
        }
        else if(active_trackers.size() == 1){
            
            //如果没有用同时出现的两个armor计算过radius,那么不用整车ckf,直接使用ekf.update/predict
            auto ball_res = calc_ballistic_one(
                (params_.response_delay + params_.shoot_delay), 
                timestamp_image, 
                *active_trackers[0],  
                timestamp,
                predict_func_double
            );
            cmd.cmd_yaw = ball_res.yaw;
            cmd.cmd_pitch = ball_res.pitch;
            cmd.last_armor_idx = active_trackers[0]->tracker_idx;
        }
        // 操作手想击打的目标enemy有 > 1个可见armor
        else{
            //算一下t_fly，用哪个板都差不多
            auto ball_res = calc_ballistic_one(
                   (params_.response_delay + params_.shoot_delay), 
                   timestamp_image, 
                   *active_trackers[0], 
                   timestamp,
                   predict_func_double
                );
            std::vector<double> yaw_armor_to_center;
            Eigen::Vector3d enemy_center_pre = enemy.enemy_ckf.predictCenterPosition(enemy.center(2), ball_res.t + params_.response_delay + params_.shoot_delay);
           
            for(size_t i = 0; i < active_trackers.size(); i++){
                auto ball_res = calc_ballistic_one(
                   (params_.response_delay + params_.shoot_delay), 
                   timestamp_image, 
                   *active_trackers[i], 
                   timestamp,
                   predict_func_double
                );
                Eigen::Vector3d armor_center_pre = FilterManage(enemy, ball_res.t + params_.response_delay + params_.shoot_delay, *active_trackers[i], timestamp);

                // 计算夹角（有符号，表示方向）[vec1 -> vec2 逆时针为正， 反之为负]
                double tmp = angleBetweenVectors(enemy_center_pre, armor_center_pre);
    
                yaw_armor_to_center.push_back(tmp);
                   
                // 低速旋转时ckf.Xe(5)即omega可用，但omega接近0时数据可能会在+/-之间跳变，需要结合last_armor_idx判断，防止频繁切换目标
                if(cmd.cmd_mode == 1){
                    bool choose_armor = false; 
                    // omega > 0 , enemy顺时针旋转
                    if(enemy.enemy_ckf.Xe(5) > 0){
                        // [-30度 ~ +45度]
                        if(yaw_armor_to_center[i] < 0.78540 && yaw_armor_to_center[i] > -0.52333){
                           cmd.cmd_yaw = ball_res.yaw;
                           cmd.cmd_pitch = ball_res.pitch;
                           cmd.booster_enable = 1;
                           cmd.last_armor_idx = active_trackers[i]->tracker_idx;
                           choose_armor = true;
                        }
                    }
                    else if(enemy.enemy_ckf.Xe(5) < 0){
                        // [-45度 ~ +30度]
                        if(yaw_armor_to_center[i] < 0.52333 && yaw_armor_to_center[i] > -0.78540){
                           cmd.cmd_yaw = ball_res.yaw;
                           cmd.cmd_pitch = ball_res.pitch;
                           cmd.booster_enable = 1;
                           cmd.last_armor_idx = active_trackers[i]->tracker_idx;
                           choose_armor = true;
                        }
                    }
                    if(!choose_armor){
                        //如果低速旋转，恰好两个armor都不在75度范围内，是不是应该让云台去准备接下一个板呢？？？
                        //保险一点的话那一小段时间控云台瞄中心？？
                        double x = enemy_center_pre.x();
                        double y = enemy_center_pre.y();
                        double z = enemy_center_pre.z();
                
                        auto ball_center = ballistic_solver_->query(x, y, z);
                        cmd.cmd_pitch = ball_center.pitch;
                        cmd.cmd_yaw = ball_center.yaw;
                        cmd.booster_enable = 1;
                        cmd.last_armor_idx = -1;
                    }
                }
            }
            // MODE = 0，意味着几乎没有旋转
            //此时已经拿到了每个armor的std::vector<double> yaw_armor_to center
            if(cmd.cmd_mode == 0){
                bool finish_choose = false;
                for(size_t i = 0; i < active_trackers.size(); i++){
                   // 对上一帧的目标armor,给一些优惠
                   if(active_trackers[i]->tracker_idx == cmd.last_armor_idx){

                      bool shoot_last = true;
 
                      for(size_t j = 0; j < yaw_armor_to_center.size(); j++){
                          // 差值 < 15度
                          if(abs(abs(yaw_armor_to_center[i]) - abs(yaw_armor_to_center[j])) > 0.2617){
                             shoot_last = false;
                          }
                      }
                      if(shoot_last){
                         auto ball_res = calc_ballistic_one(
                            (params_.response_delay + params_.shoot_delay), 
                            timestamp_image, 
                            *active_trackers[i], 
                            timestamp,
                            predict_func_double
                         );
                         cmd.cmd_yaw = ball_res.yaw;
                         cmd.cmd_pitch = ball_res.pitch;
                         cmd.booster_enable = 1;
                         cmd.last_armor_idx = active_trackers[i]->tracker_idx;
                         finish_choose = true;
                         break;
                      }
                  }
               }
               //出了上面的循环，要么没有这一帧的active_trackers没有上一帧的目标，要么相差超过临界，此时大公无私地选击打目标armor了
               //平动至少会瞄一个
               if(!finish_choose){
                  double yaw_min = 180.0;
                  int aim_idx = -1;
                  // 瞄最正对的armor
                  for(size_t k = 0; k < yaw_armor_to_center.size(); k++){
                     if(yaw_armor_to_center[k] < yaw_min){
                        yaw_min = yaw_armor_to_center[k];
                        aim_idx = k;
                     }
                  }
                  auto ball_res = calc_ballistic_one(
                          (params_.response_delay + params_.shoot_delay), 
                          timestamp_image, 
                          *active_trackers[aim_idx], 
                          timestamp,
                          predict_func_double
                       );
                   cmd.cmd_yaw = ball_res.yaw;
                   cmd.cmd_pitch = ball_res.pitch;
                   cmd.booster_enable = 1;
                   cmd.last_armor_idx = active_trackers[aim_idx]->tracker_idx;
               }
            }
        }
    }
}
BallisticResult EnemyPredictor::calc_ballistic_one
            (double delay, rclcpp::Time timestamp_image, ArmorTracker& tracker, double timestamp,
                std::function<Eigen::Vector3d(ArmorTracker&, double, double)> _predict_func)
{

    BallisticResult ball_res;
    Eigen::Vector3d predict_pos_base_link;
    //Eigen::Isometry3d base_link2gimbal_transform = getTrans("base_link", "gimbal", timestamp_image);
    double t_fly = 0.0;  // 飞行时间（迭代求解）

    rclcpp::Time tick = node_->now();

    for (int i = 0; i < 6; ++i) {
        rclcpp::Time tock = node_->now();

        rclcpp::Duration elapsed = tock - tick;
        
        double latency = delay + elapsed.seconds();

        double code_dt = tock.seconds() - timestamp_image.seconds();
        //RCLCPP_INFO(this->get_logger(), "latency time: %.6f", latency);
        predict_pos_base_link = _predict_func(tracker, t_fly + latency + code_dt, timestamp);

        double x = predict_pos_base_link.x();
        double y = predict_pos_base_link.y();
        double z = predict_pos_base_link.z();

        ball_res = ballistic_solver_->query(x, y, z);

        if (!ball_res.success) {
            RCLCPP_WARN(get_logger(), "[calc Ballistic] too far to hit it\n");
            return ball_res;
        }
        t_fly = ball_res.t;
    } 
    // 考虑自身z的变化
    //Eigen::Vector3d z_vec = Eigen::Vector3d::Zero();
    // address it later!!!!!!!!!!!!!!!!!!!
    //z_vec << 0, 0, cmd.robot.z_velocity * (params_.shoot_delay + t_fly);

    //ball_res = bac.final_ballistic(base_link2gimbal_transform, predict_pos_base_link - z_vec);
    //RCLCPP_DEBUG(get_logger(), "calc_ballistic: predict_pos_base_link: %f %f %f", predict_pos_base_link(0), predict_pos_base_link(1), predict_pos_base_link(2));
    return ball_res;
}
BallisticResult EnemyPredictor::calc_ballistic_second
            (double delay, rclcpp::Time timestamp_image, double timestamp, int phase_id, Enemy& enemy,
                std::function<Eigen::Vector3d(Enemy&, double, int)> _predict_func)
{

    BallisticResult ball_res;
    Eigen::Vector3d predict_pos_base_link;
    //Eigen::Isometry3d base_link2gimbal_transform = getTrans("base_link", "gimbal", timestamp_image);
    double t_fly = 0.0;  // 飞行时间（迭代求解）

    rclcpp::Time tick = node_->now();

    for (int i = 0; i < 6; ++i) {
        rclcpp::Time tock = node_->now();

        rclcpp::Duration elapsed = tock - tick;
        
        double latency = delay + elapsed.seconds();

        double code_dt = tock.seconds() - timestamp_image.seconds();

        predict_pos_base_link = _predict_func(enemy, t_fly + latency + code_dt, phase_id);

        double x = predict_pos_base_link.x();
        double y = predict_pos_base_link.y();
        double z = predict_pos_base_link.z();

        ball_res = ballistic_solver_->query(x, y, z);


        if (!ball_res.success) {
            RCLCPP_WARN(get_logger(), "[calc Ballistic] too far to hit it\n");
            return ball_res;
        }
        t_fly = ball_res.t;
    } 
    // 考虑自身z的变化
    //Eigen::Vector3d z_vec = Eigen::Vector3d::Zero();
    // address it later!!!!!!!!!!!!!!!!!!!
    //z_vec << 0, 0, cmd.robot.z_velocity * (params_.shoot_delay + t_fly);

    //ball_res = bac.final_ballistic(base_link2gimbal_transform, predict_pos_base_link - z_vec);
    //RCLCPP_DEBUG(get_logger(), "calc_ballistic: predict_pos_base_link: %f %f %f", predict_pos_base_link(0), predict_pos_base_link(1), predict_pos_base_link(2));
    return ball_res;
}
Eigen::Vector3d EnemyPredictor::FilterManage(Enemy &enemy, double dt, ArmorTracker& tracker, double timestamp){
    
    Eigen::Vector3d xyyaw_pre_ekf = tracker.ekf.predict_position(dt);
    ZEKF::Vz z_pre = tracker.zekf.predict_position(dt);              // z的处理后面再调，用z_ekf or 均值滤波
    Eigen::Vector3d xyz_ekf_pre = Eigen::Vector3d(xyyaw_pre_ekf[0], xyyaw_pre_ekf[1], z_pre[0]);
    
    //visualizeAimCenter(xyz_ekf_pre, cv::Scalar(0, 255, 0));
    geometry_msgs::msg::Pose pose_pre_ekf = vectorToPose(xyz_ekf_pre);
    predictor_node->annotate(pose_pre_ekf);
    
    //if(!enemy.radius_cal){
    //   return xyz_ekf_pre;
    //}
    Eigen::Vector3d xyz_pre_ckf = enemy.enemy_ckf.predictArmorPosition(enemy.center(2), tracker.phase_id, dt);
   
    Eigen::Vector3d enemy_xyz = Eigen::Vector3d(enemy.enemy_ckf.Xe(0),enemy.enemy_ckf.Xe(2), z_pre[0]);

    //for(int i = 0; i < 8; i++){
    //    RCLCPP_INFO(get_logger(), "CKF: Xe(%d) = %lf", i, enemy.enemy_ckf.Xe(i));
    //}
   
    //visualizeAimCenter(xyz_pre_ckf, cv::Scalar(255, 0, 0));
    //visualizeAimCenter(enemy_xyz, cv::Scalar(0, 255, 255));   // For DEBUG
    geometry_msgs::msg::Pose center = vectorToPose(enemy_xyz);
    predictor_node->annotate(center);

    geometry_msgs::msg::Pose pose_pre_ckf = vectorToPose(xyz_pre_ckf);
    predictor_node->annotate(pose_pre_ckf);
    
    double k = 1.0;
    double r0 = 1.0;

    double v_x = tracker.ekf.Xe(3);
    double v_y = tracker.ekf.Xe(4);
    double v = std::sqrt(v_x * v_x + v_y * v_y);
    double V = std::max(v, 0.01);

    double omega = tracker.ekf.Xe(5);

    double r = std::abs(omega)/V;

    // exp(k * (r/r0 - 1))
    double exponent = k * (r / r0 - 1.0);
    double exp_value = std::exp(exponent);
    
    // 分母: 1 + exp(k * (r/r0 - 1))
    double denominator = 1.0 + exp_value;
    
    //EKF权重系数: 1 / (1 + exp(k * (r/r0 - 1)))
    double w_ekf = 1.0 / denominator;
    
    // 5. 计算CKF权重系数: exp(k * (r/r0 - 1)) / (1 + exp(k * (r/r0 - 1)))
    double w_ckf = exp_value / denominator;
    
    Eigen::Vector3d fusion_pre = w_ekf * xyz_ekf_pre + w_ckf * xyz_pre_ckf;
    
    //visualizeAimCenter(fusion_pre, cv::Scalar(0, 0, 255));
    geometry_msgs::msg::Pose fusion_pose = vectorToPose(fusion_pre);
    predictor_node->annotate(fusion_pose);

    // =================== 添加EKF滤波器可视化到rviz ===================
    visualization_msgs::msg::Marker ekf_armor_marker;
    ekf_armor_marker.header.frame_id = "base_link";
    ekf_armor_marker.header.stamp = node_->now();
    ekf_armor_marker.ns = "filter_results";
    ekf_armor_marker.id = enemy.class_id * 100 + 3;  // EKF使用不同的ID
    
    ekf_armor_marker.type = visualization_msgs::msg::Marker::SPHERE;
    ekf_armor_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ekf_armor_marker.pose.position.x = xyz_ekf_pre.x();
    ekf_armor_marker.pose.position.y = xyz_ekf_pre.y();
    ekf_armor_marker.pose.position.z = xyz_ekf_pre.z();
    ekf_armor_marker.pose.orientation.w = 1.0;
    
    ekf_armor_marker.scale.x = 0.06;  // 装甲板尺寸
    ekf_armor_marker.scale.y = 0.06;
    ekf_armor_marker.scale.z = 0.06;
    
    // 设置颜色：绿色表示EKF预测的装甲板位置
    ekf_armor_marker.color.r = 0.0;
    ekf_armor_marker.color.g = 1.0;
    ekf_armor_marker.color.b = 0.0;
    ekf_armor_marker.color.a = 0.9;
    
    ekf_armor_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(ekf_armor_marker);
    
    // 1. 可视化CKF估计的敌人中心（enemy_xyz）
    visualization_msgs::msg::Marker ckf_center_marker;
    ckf_center_marker.header.frame_id = "base_link";  // 与之前的坐标系一致
    ckf_center_marker.header.stamp = node_->now();
    ckf_center_marker.ns = "filter_results";
    ckf_center_marker.id = enemy.class_id * 100 + 1;  // 使用不同的ID范围
    
    ckf_center_marker.type = visualization_msgs::msg::Marker::SPHERE;
    ckf_center_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ckf_center_marker.pose.position.x = enemy_xyz.x();
    ckf_center_marker.pose.position.y = enemy_xyz.y();
    ckf_center_marker.pose.position.z = enemy_xyz.z();
    ckf_center_marker.pose.orientation.w = 1.0;
    
    ckf_center_marker.scale.x = 0.08;  // 比实际敌人中心小一点
    ckf_center_marker.scale.y = 0.08;
    ckf_center_marker.scale.z = 0.08;
    
    // 设置颜色：黄色表示CKF估计的敌人中心
    ckf_center_marker.color.r = 1.0;
    ckf_center_marker.color.g = 1.0;
    ckf_center_marker.color.b = 0.0;
    ckf_center_marker.color.a = 0.9;
    
    ckf_center_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(ckf_center_marker);
    
    visualization_msgs::msg::Marker ckf_armor_marker;
    ckf_armor_marker.header.frame_id = "base_link";
    ckf_armor_marker.header.stamp = node_->now();
    ckf_armor_marker.ns = "filter_results";
    ckf_armor_marker.id = enemy.class_id * 100 + 2;  // 不同ID
    
    ckf_armor_marker.type = visualization_msgs::msg::Marker::SPHERE;
    ckf_armor_marker.action = visualization_msgs::msg::Marker::ADD;
    
    ckf_armor_marker.pose.position.x = xyz_pre_ckf.x();
    ckf_armor_marker.pose.position.y = xyz_pre_ckf.y();
    ckf_armor_marker.pose.position.z = xyz_pre_ckf.z();
    ckf_armor_marker.pose.orientation.w = 1.0;
    
    ckf_armor_marker.scale.x = 0.06;  // 装甲板尺寸比中心小
    ckf_armor_marker.scale.y = 0.06;
    ckf_armor_marker.scale.z = 0.06;
    
    // 蓝色表示CKF预测的装甲板位置
    ckf_armor_marker.color.r = 0.0;
    ckf_armor_marker.color.g = 0.0;
    ckf_armor_marker.color.b = 1.0;  // 纯蓝色
    ckf_armor_marker.color.a = 0.9;
    
    ckf_armor_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    enemy_markers_.markers.push_back(ckf_armor_marker);

    //visualization_msgs::msg::Marker fusion_marker;
    //fusion_marker.header.frame_id = "base_link";
    //fusion_marker.header.stamp = this->now();
    //fusion_marker.ns = "fusion_results";
    //fusion_marker.id = enemy.class_id * 100 + 4;  // 新的ID，确保不冲突
    //
    //fusion_marker.type = visualization_msgs::msg::Marker::SPHERE;
    //fusion_marker.action = visualization_msgs::msg::Marker::ADD;
    //
    //fusion_marker.pose.position.x = fusion_pre.x();
    //fusion_marker.pose.position.y = fusion_pre.y();
    //fusion_marker.pose.position.z = fusion_pre.z();
    //fusion_marker.pose.orientation.w = 1.0;
    //
    //// 设置稍大的尺寸以突出显示融合结果
    //fusion_marker.scale.x = 0.07;
    //fusion_marker.scale.y = 0.07;
    //fusion_marker.scale.z = 0.07;
    //
    //// 设置颜色：红色表示融合结果
    //fusion_marker.color.r = 1.0;  // 红色
    //fusion_marker.color.g = 0.0;
    //fusion_marker.color.b = 0.0;
    //fusion_marker.color.a = 1.0;  // 不透明度设为1.0，使其更显眼
    //
    //fusion_marker.lifetime = rclcpp::Duration::from_seconds(0.2);
    //enemy_markers_.markers.push_back(fusion_marker);
   
    return xyz_pre_ckf;

}
//--------------------------------------------TOOL----------------------------------------------------------------

void EnemyPredictor::create_new_tracker(const Detection &detection,double timestamp, std::vector<int>& active_armor_idx) {

    ArmorTracker new_tracker;
    new_tracker.position = detection.position;
    new_tracker.last_position = detection.position;
    new_tracker.yaw = detection.yaw;
    new_tracker.area_2d = detection.area_2d;
    new_tracker.armor_class_id = detection.armor_class_id;
    new_tracker.tracker_idx = detection.armor_idx;
    new_tracker.is_active = true;
    new_tracker.ekf.init(detection.position, timestamp);
    new_tracker.zekf.init(detection.position(2), timestamp);
    
    enemies_[new_tracker.armor_class_id % 10 - 1].is_active = true;

    active_armor_idx.push_back(armor_trackers_.size()); // 先后push_back的顺序
    
    armor_trackers_.push_back(new_tracker);
}


double EnemyPredictor::angleBetweenVectors(Eigen::Vector3d vec1, Eigen::Vector3d vec2){
    
    Eigen::Vector2d v1(-vec1(0), -vec1(1));      // enemy_center -> 原点
    Eigen::Vector2d v2(vec2(0) - vec1(0),
                         vec2(1) - vec1(1));      // enemy_center -> armor_center
    
    // 计算点积和叉积
    double dot = v1.dot(v2);
    double cross = v1.x() * v2.y() - v1.y() * v2.x();  // 叉积的z分量
    // 计算夹角（有符号，表示方向）[vec1 -> vec2 逆时针为正， 反之为负]
    double angle = std::atan2(cross, dot);
    return angle;
}
double EnemyPredictor::normalize_angle(double angle) {
    while (angle > M_PI) angle -= 2 * M_PI;
    while (angle <= -M_PI) angle += 2 * M_PI;
    return angle;
}

double EnemyPredictor::angle_difference(double a, double b) {
    double diff = normalize_angle(a - b);
    if (diff > M_PI) diff -= 2 * M_PI;
    return diff;
}
double EnemyPredictor::getYawfromQuaternion(double w, double x, double  y, double z) {

    Eigen::Quaterniond q = Eigen::Quaterniond(w, x, y, z);
    Eigen::Quaterniond q_norm = q.normalized();
    Eigen::Matrix3d R = q_norm.toRotationMatrix();
    
    Eigen::Vector3d euler;
    const double eps = 1e-6;
    
    // 检查万向节锁
    if (std::abs(R(2, 0)) < 1.0 - eps) {
        // 正常情况
        euler(1) = -std::asin(R(2, 0));  // pitch
        euler(0) = std::atan2(R(2, 1), R(2, 2));  // roll
        euler(2) = std::atan2(R(1, 0), R(0, 0));  // yaw
    } else {
        // 万向节锁情况
        euler(1) = -std::copysign(M_PI/2, R(2, 0));  // pitch = ±90°
        euler(0) = std::atan2(-R(0, 1), R(1, 1));  // roll
        euler(2) = 0;  // yaw设为0
    }
    
    return euler(2);  // 只返回yaw
}

// 可视化：aim center
void EnemyPredictor::visualizeAimCenter(const Eigen::Vector3d& armor_base_link, 
                                           const cv::Scalar& point_color) {
    if (visualize_.armor_img.empty()) {
        RCLCPP_WARN(get_logger(), "armor_img is empty, skipping visualization");
        return;
    }
    
    // 检查图像是否有效
    if (visualize_.armor_img.cols <= 0 || visualize_.armor_img.rows <= 0) {
        RCLCPP_WARN(get_logger(), "armor_img has invalid size, skipping visualization");
        return;
    }
    try{
            // 1. 将base_link系坐标转换到相机系
    Eigen::Vector3d aim_center_cam = visualize_.camara_to_base_link.inverse() * armor_base_link;
    
    // 2. 准备3D点（相机坐标系下）
    std::vector<cv::Point3d> object_points;
    object_points.push_back(cv::Point3d(
        aim_center_cam.x(), 
        aim_center_cam.y(), 
        aim_center_cam.z()
    ));

    // 3. 投影时使用零旋转和零平移（因为点已经在相机坐标系中）
    cv::Mat zero_rvec = cv::Mat::zeros(3, 1, CV_64F);  // 零旋转
    cv::Mat zero_tvec = cv::Mat::zeros(3, 1, CV_64F);  // 零平移

    // 4. 投影3D点到2D图像平面
    std::vector<cv::Point2d> reprojected_points;
    cv::projectPoints(object_points,
                      zero_rvec,                      // 零旋转
                      zero_tvec,                      // 零平移
                      visualize_.camera_matrix,       // 相机内参矩阵
                      visualize_.dist_coeffs,         // 畸变系数
                      reprojected_points);
    
    // 在访问vector前检查
    if (reprojected_points.empty()) {
        RCLCPP_WARN(get_logger(), "No reprojected points, skipping visualization");
        return;
    }
    
    cv::Point2d center = reprojected_points[0];
    
    // 检查坐标是否有效（不是NaN或INF）
    if (std::isnan(center.x) || std::isnan(center.y) || 
        std::isinf(center.x) || std::isinf(center.y)) {
        RCLCPP_WARN(get_logger(), "Invalid projected coordinates: (%.1f, %.1f)", 
                   center.x, center.y);
        return;
    }

    // 6. 在图像上绘制点
    if (!reprojected_points.empty() && !visualize_.armor_img.empty()) {
        cv::Point2d center = reprojected_points[0];
        
        // 检查点是否在图像范围内
        if (center.x >= 0 && center.x < visualize_.armor_img.cols &&
            center.y >= 0 && center.y < visualize_.armor_img.rows) {
            
            // 绘制指定颜色的圆点
            cv::circle(visualize_.armor_img, center, 5, point_color, -1);   // 实心圆
            
        } else {
            RCLCPP_WARN(get_logger(), 
                       "Projected point (%.1f, %.1f) out of image bounds [%d x %d]", 
                       center.x, center.y, visualize_.armor_img.cols, visualize_.armor_img.rows);
        }
    } else {
        RCLCPP_WARN(get_logger(), "No projected points or empty image");
    }
    } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(get_logger(), 
                    "Cv Project:"  << e.what());
    }
   
}
geometry_msgs::msg::Pose EnemyPredictor::vectorToPose(const Eigen::Vector3d& point) {
    geometry_msgs::msg::Pose pose;
    
    // 设置位置
    pose.position.x = point.x();
    pose.position.y = point.y();  // 注意：原函数中y被用作深度
    pose.position.z = point.z();
    
    // 设置姿态为单位四元数（无旋转）
    pose.orientation.x = 0.0;
    pose.orientation.y = 0.0;
    pose.orientation.z = 0.0;
    pose.orientation.w = 1.0;
    
    return pose;
}

