#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <rclcpp/node.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <message_filters/subscriber.hpp>
#include <message_filters/cache.hpp>

#include <rm_msgs/msg/imu_stamped.hpp>
#include <rm_msgs/msg/state.hpp>
#include <rm_msgs/msg/control.hpp>
#include <rm_msgs/srv/set_vision_mode.hpp>
#include <rm_msgs/srv/toggle_vision_follow_id.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <foxglove_msgs/msg/color.hpp>
#include <foxglove_msgs/msg/image_annotations.hpp>
#include <foxglove_msgs/msg/point2.hpp>

#include "basic_predictor/ballestic_utils.hpp"

class UniformSpeedManager {
public:
    explicit UniformSpeedManager() : roll(std::numeric_limits<double>::infinity()), roll_variation(0.) {

    }

    void update(const geometry_msgs::msg::Pose& pose) {
        double meas = pose.orientation.x; // measured roll
        if (roll == std::numeric_limits<double>::infinity()) {
            roll = std::remainder(meas, M_PI*2/5);
            roll_variation = 0.;
        } else {
            double diff = std::remainder(meas-roll, M_PI*2/5);
            roll = roll + diff;
            roll_variation = roll_variation + diff;
        }
    }

    geometry_msgs::msg::Pose predict(const geometry_msgs::msg::Pose& pose, double dt) {
        double orient = 0.;
        if (std::fabs(roll_variation) > M_PI / 5) {
            orient = (roll_variation > 0) ? 1. : -1.;
        } else {
            return pose;
        }

        tf2::Quaternion q;
        q.setRPY(pose.orientation.x + M_PI/3*orient*dt, pose.orientation.y, pose.orientation.z);
        
        auto target = geometry_msgs::msg::Pose(pose
                ).set__orientation(tf2::toMsg(q));

        Eigen::Isometry3d transform;
        tf2::fromMsg(target, transform);

        Eigen::Vector3d target_in_world(0.70, 0., 0.);
        target.set__position(tf2::toMsg(transform * target_in_world));

        return target;
    }

    double roll;
    double roll_variation;
};


class PredictorNodeInterface: public rclcpp::Node {
public:
    PredictorNodeInterface();

    rm_msgs::msg::Imu get_imu(rclcpp::Time stamp = rclcpp::Time(0, 0, RCL_ROS_TIME));
    rm_msgs::msg::State get_robot_state();

    uint8_t get_vision_follow_id();

    BallisticResult solve_ballistic(geometry_msgs::msg::Point point);

    void send_command(const rm_msgs::msg::Imu& imu, uint8_t control_mode, bool booster_enable);

    void annotate(const std::vector<geometry_msgs::msg::Pose>& poses);
    void annotate(const geometry_msgs::msg::Pose& poses);

    std::shared_ptr<std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray>)>>
    register_callback(const std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray>)>& callback);

private:
    static constexpr const char* kNodeName = "basic_predictor";

    // polymorphism
    std::unique_ptr<PredictorInterface> predictor;

    // callback handle
    std::weak_ptr<std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray>)> vision_callback;

    // message cache
    std::shared_ptr<sensor_msgs::msg::CameraInfo> camera_info;
    std::shared_ptr<message_filters::Cache<rm_msgs::msg::ImuStamped>> imu_stamped_cache;
    std::shared_ptr<rm_msgs::msg::State> robot_state;
    std::shared_ptr<foxglove_msgs::msg::ImageAnnotations> image_annotations;
    uint8_t vision_follow_id;

    // publishers
    std::shared_ptr<rclcpp::Publisher<rm_msgs::msg::Control>> control_pub;
    std::shared_ptr<rclcpp::Publisher<foxglove_msgs::msg::ImageAnnotations>> image_annotations_pub;

    // subscriptions
    std::shared_ptr<rclcpp::Subscription<sensor_msgs::msg::CameraInfo>> camera_info_sub;
    std::shared_ptr<rclcpp::Subscription<rm_msgs::msg::State>> state_sub;
    std::shared_ptr<rclcpp::Subscription<vision_msgs::msg::Detection2DArray>> vision_sub;
    std::shared_ptr<rclcpp::Subscription<rm_msgs::msg::Control>> ext_control_sub;

    // services
    std::shared_ptr<rclcpp::Service<rm_msgs::srv::SetVisionMode>> set_vision_mode_srv;
    std::shared_ptr<rclcpp::Service<rm_msgs::srv::ToggleVisionFollowId>> toggle_vision_follow_id_srv;

    // clients
    std::shared_ptr<rclcpp::Client<rm_msgs::srv::ToggleVisionFollowId>> toggle_vision_follow_id_cli;

    // ballestic solver
    /// NOTE: To be honest, I dont want to couple ballestic and node functions but I have to.
    std::shared_ptr<BallisticSolver> ballestic_solver;
};

class PredictorInterface {
public:
    explicit PredictorInterface(PredictorNodeInterface* node);
    virtual ~PredictorInterface() noexcept = default;
    PredictorNodeInterface* node;
};

class EnemyPredictor: public PredictorInterface {
public:
    explicit EnemyPredictor(PredictorNodeInterface* node);
};

class OutpostPredictor: public PredictorInterface {
public:
    explicit OutpostPredictor(PredictorNodeInterface* node);
};

class EnergyUniformSpeedPredictor: public PredictorInterface {
public:
    explicit EnergyUniformSpeedPredictor(PredictorNodeInterface* node);



    void predict(const vision_msgs::msg::Detection2DArray &msg);

    UniformSpeedManager manager;
};

class EnergyVariableSpeedPredictor: public PredictorInterface {
public:
    explicit EnergyVariableSpeedPredictor(PredictorNodeInterface* node);
};
