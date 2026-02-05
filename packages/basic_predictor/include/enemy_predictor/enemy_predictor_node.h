#ifndef ENEMY_PREDICTOR_NODE_H  
#define ENEMY_PREDICTOR_NODE_H  

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <memory>
#include <cmath>
#include "rclcpp/rclcpp.hpp"
#include "Eigen/Dense"
#include <opencv2/opencv.hpp>
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <opencv2/opencv.hpp>
#include "enemy_predictor/armor_filter.h"
#include "enemy_predictor/enemy_filter.h"

#include <sensor_msgs/msg/image.hpp>
#include "datatypes.h"
#include "image_transport/image_transport.hpp"
#include <vision_msgs/msg/detection2_d_array.hpp>
#include "rm_msgs/msg/rm_robot.hpp"
#include "rm_msgs/msg/control.hpp"
#include "visualization_msgs/msg/marker_array.hpp"
#include "geometry_msgs/msg/point.hpp"
#include "enemy_predictor/node_interface.hpp"


class EnemyPredictor: public PredictorInterface {

public:    
    explicit EnemyPredictor(rclcpp::Node* node);
    virtual ~EnemyPredictor() = default;

    rclcpp::Logger get_logger() const {
        if (!node_) {
            throw std::runtime_error("Node pointer is null");
        }
        return node_->get_logger();
    }

    static constexpr size_t MAX_ENEMIES = 8;

    struct TF{
        Eigen::Isometry3d odom_to_gimbal;
        Eigen::Isometry3d camara_to_odom;
    }tf_;
    struct Detection {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        Eigen::Vector3d position;
        Eigen::Quaterniond orientation; 
        int armor_class_id;         
        int armor_idx; 
        double yaw;    
        double area_2d;      
    
        Detection() = default;
        
        Detection(const Eigen::Vector3d& pos, int armor_class_id_, int armor_idx_, double y = 0)
            : position(pos), armor_class_id(armor_class_id_),armor_idx(armor_idx_),yaw(y) {}
    };
    // 装甲板跟踪器
    struct ArmorTracker {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        
        bool is_active = true;
        int tracker_idx;
        int armor_class_id;
  
        int missing_frames = 0;
        int phase_id_cnt = 0;
        
        Eigen::Vector3d position;
        Eigen::Vector3d last_position;
        Eigen::Vector3d predicted_position;
        std::vector<Eigen::Vector3d> position_history;
        
        ArmorXYYAWEKF ekf;
        ZEKF zekf;
        
        double area_2d = 0.0;
        
        double last_update_time = 0.0;
        
        int assigned_enemy_idx = -1;
        int phase_id = -1;
        //double phase_conf = 0.0;
        
        // 朝向相关
        double yaw = 0.0;
        double last_yaw = 0.0;
        int yaw_round = 0;

        ArmorTracker() = default;
        
        ArmorTracker(int armor_idx, int armor_class_id,
                    const Eigen::Vector3d& init_pos, double timestamp,
                    double armor_yaw = 0.0, double area_2d = 0.0);
        
        void update(const Eigen::Vector3d& new_position, 
                   int armor_class_id, double timestamp,
                   double armor_yaw = 0.0);
        
        std::string get_string_id() const;
    };
    
    
    // 敌人结构体
    struct Enemy {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        int class_id = -1;
        std::vector<int> armor_tracker_ids;
        Eigen::Vector3d center = Eigen::Vector3d::Zero();
        std::vector<double> radius{0.25, 0.25};
        bool radius_cal = false;
        bool is_active = false;
        bool is_valid = false;
        int missing_frame = 0;

        EnemyCKF enemy_ckf;
        
        Enemy() = default; 

        void reset(){
           center = Eigen::Vector3d::Zero();
           radius_cal = false;
           is_active = false;
           is_valid = false;
           missing_frame = 0;
           enemy_ckf.initializeCKF();
           std::vector<double> radius{0.25, 0.25}; // 保留历史数据，越打越准可行吗？？
        }
    };
    struct Command{
        double high_spd_rotate_thresh = 0.0;
        double rotate_thresh = 0.0;
        double yaw_thresh = 0.0;
        double cmd_pitch = 0.0;
        double cmd_yaw = 0.0;
        int last_target_enemy_idx = -1;
        int target_enemy_idx = -1;
        bool right_press = false;
        int cmd_mode; //  0 -> 平动 , 1 -> 小陀螺
        int last_armor_idx = -1;
        bool booster_enable = 0;
        std::vector<double> stored_yaw_offsets{};
        std::vector<double> stored_pitch_offsets{};
    }cmd;
    struct EnemyPredictorNodeParams{
        
        std::string target_frame;
        std::string camera_name;
        bool right_press;  // 按下右键
        // 火控参数
        double change_armor_time_thresh;
        double dis_yaw_thresh;
        double gimbal_error_dis_thresh;            // 自动发弹阈值，限制云台误差的球面意义距离
        double pitch_error_dis_thresh;             // 自动发弹阈值，目标上下小陀螺时pitch限制
        // 延迟参数
        double response_delay;  // 系统延迟(程序+通信+云台响应)
        double shoot_delay;     // 发弹延迟
    
        double pitch_offset_high_hit_low;  //deg
        double pitch_offset_low_hit_high;  //deg
    }params_;    

    // 数据容器
    ArmorTracker armor_tracker;
    std::vector<ArmorTracker> armor_trackers_;
    std::array<Enemy, MAX_ENEMIES> enemies_;
 
    double timestamp;
    
    // 可视化相关
    struct VisualizeData {
        cv::Mat armor_img{};
        cv::Mat camera_matrix{};
        cv::Mat dist_coeffs{};
        cv::Mat camera_rvec = cv::Mat::zeros(3, 1, CV_64F);
        cv::Mat camera_tvec = cv::Mat::zeros(3, 1, CV_64F);;
        Eigen::Isometry3d camara_to_odom{};
        Eigen::Vector3d pos_camera{};
        cv::Point2f camera_heart{};
        std::string image_frame;

    } visualize_;
    std::vector<cv::Point3f> small_object_points = {
        {-0.0675, 0.0275, 0.},
        {-0.0675, -0.0275, 0.},
        {0.0675, -0.0275, 0.},
        {0.0675, 0.0275, 0.}
    };

    std::vector<cv::Point3f> large_object_points = {
        {-0.115, 0.029, 0.},
        {-0.115, -0.029, 0.},
        {0.115, -0.029, 0.},
        {0.115, 0.029, 0.}
    };    
public:
    void updateArmorDetection(std::vector<cv::Point3f> object_points,
                              Detection& det,
                              rclcpp::Time timestamp_det);

    void ToupdateArmors(const std::vector<Detection, Eigen::aligned_allocator<Detection>>& detections,
                     double timestamp, std::vector<int>& active_armor_idx);
    
    Eigen::Isometry3d getTrans(const std::string& source_frame, 
                               const std::string& target_frame,
                               rclcpp::Time timestamp_det);
    void EnemyManage(double timestamp, rclcpp::Time timestamp_det, std::vector<int>& active_enemies_idx, std::vector<int>& active_armor_idx);
private:
    void initBallistic();
    void initFilterParams();
    void initCommandParams();
    rclcpp::Node* node_;

    // 敌人分配和更新
    void updateEnemy(Enemy& enemy, double timestamp, std::vector<int>& active_armor_idx);

    void updateSingleEnemy(Enemy& enemy, double timestamp);
    void calculateEnemyCenterAndRadius(Enemy& enemy, double timestamp, std::vector<ArmorTracker*> active_armors_this_enemy);
    Eigen::Vector3d FilterManage(Enemy &enemy, double dt, ArmorTracker& tracker, double timestamp);
    // 相位处理
    //void updateArmorPhase(Enemy& enemy, ArmorTracker& tracker, double timestamp);
    void findBestPhaseForEnemy(Enemy& enemy, ArmorTracker& tracker,  std::vector<ArmorTracker*> active_armors_this_enemy);
    int estimatePhaseFromPosition(const Enemy& enemy, const ArmorTracker& tracker);
    // 决策 + 弹道
    BallisticResult calc_ballistic_one(double delay, rclcpp::Time timestamp_det, ArmorTracker& tracker, double timestamp, std::function<Eigen::Vector3d(ArmorTracker&, double, double)> _predict_func);
    BallisticResult calc_ballistic_second(double delay, rclcpp::Time timestamp_det, double timestamp, int phase_id, Enemy& enemy, std::function<Eigen::Vector3d(Enemy&, double, int)> _predict_func);
    void getCommand(Enemy& enemy, double timestamp, rclcpp::Time timestamp_det, std::vector<int>& active_armor_idx);
    int ChooseMode(Enemy &enemy, double timestamp);
    // tool
    void create_new_tracker(const Detection &detection, double timestamp, std::vector<int>& active_armor_idx);

    void useGeometricCenterSimple(Enemy& enemy, 
                                const std::vector<ArmorTracker*>& active_armors);

    // 角度处理
    double angleBetweenVectors(Eigen::Vector3d vec1, Eigen::Vector3d vec2);
    double normalize_angle(double angle);
    double angle_difference(double a, double b);

    double getYawfromQuaternion(double w, double x, double  y, double z);
    
    // 可视化
    void visualizeAimCenter(const Eigen::Vector3d& armor_odom, const cv::Scalar& point_color = cv::Scalar(0, 0, 255));
    
    rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detector_sub;

    rclcpp::Publisher<rm_msgs::msg::Control>::SharedPtr control_pub;

    rm_msgs::msg::Control::SharedPtr control_msg;

    std::shared_ptr<tf2_ros::Buffer> tf2_buffer;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener;
    
    rclcpp::Time time_det;
    rclcpp::Time time_image;

    std::shared_ptr<BallisticSolver> ballistic_solver_;

    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr enemy_markers_pub_;
    visualization_msgs::msg::MarkerArray enemy_markers_;

    void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr detection_msg);

};
#include "rclcpp_components/register_node_macro.hpp"
//RCLCPP_COMPONENTS_REGISTER_NODE(EnemyPredictor)  // 注册插件


#endif // _ENEMY_PREDICTOR_NODE_H