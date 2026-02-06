#include "basic_predictor/node_interface.hpp"

#include <std_msgs/msg/header.hpp>
#include "node_interface.hpp"

PredictorInterface::PredictorInterface(PredictorNodeInterface* node) : node(node) {
}

EnemyPredictor::EnemyPredictor(PredictorNodeInterface* node) : PredictorInterface(node) {
}

OutpostPredictor::OutpostPredictor(PredictorNodeInterface* node) : PredictorInterface(node) {
}

EnergyUniformSpeedPredictor::EnergyUniformSpeedPredictor(PredictorNodeInterface* node) : PredictorInterface(node) {

}

void EnergyUniformSpeedPredictor::predict(const vision_msgs::msg::Detection2DArray& msg) {
    auto best_det = *std::max_element(
    msg.detections.begin(), msg.detections.end(),
    [](const auto& a, const auto& b) {
        auto best_score = [](const vision_msgs::msg::Detection2D& d) {
            double s = -std::numeric_limits<double>::infinity();
            for (const auto& r : d.results) {
                s = std::max<double>(s, r.hypothesis.score);
            }
            return s;
        };
        return best_score(a) < best_score(b);
    });

    if (best_det.results.empty()) {
        node->send_command(
            node->get_robot_state().imu,
            0,
            0
        );
        return;
    }

    auto target = best_det.results[0].pose.pose;
    manager.update(target);

    auto bac_res = node->solve_ballistic(target.position);
    for (size_t i = 0; i < node->get_parameter("ballestic.iteration").as_int(); i++) {
        target = manager.predict(target, bac_res.fly_time);
        bac_res = node->solve_ballistic(target.position);
    }

    auto diff = std::remainder(target.orientation.x - roll, M_PI*2/5);
    roll = target.orientation.x;

    stamp = msg.header.stamp;
    if (diff > M_PI / 5) {
        toggle_stamp = msg.header.stamp;
    }

    if ((rclcpp::Time(stamp) - rclcpp::Time(toggle_stamp)).nanoseconds <= 3e8
        and (rclcpp::Time(msg.header.stamp) - rclcpp::Time(toggle_stamp)).nanoseconds > 3e8
            and manager.is_initialized()) {
        node->send_command(
            rm_msgs::msg::Imu().set__roll(0.).set__pitch(bac_res.pitch).set__yaw(bac_res.yaw),
            1,
            1
        );
    } else {
        node->send_command(
            rm_msgs::msg::Imu().set__roll(0.).set__pitch(bac_res.pitch).set__yaw(bac_res.yaw),
            1,
            0 
        );
    }
}

void EnergyVariableSpeedPredictor::predict(const vision_msgs::msg::Detection2DArray& msg) {
    auto sorted_msg = msg;
    std::sort(
    sorted_msg.detections.begin(), sorted_msg.detections.end(),
    [](const auto& a, const auto& b) {
        auto best_score = [](const vision_msgs::msg::Detection2D& d) {
            double s = -std::numeric_limits<double>::infinity();
            for (const auto& r : d.results) {
                s = std::max<double>(s, r.hypothesis.score);
            }
            return s;
        };
        return best_score(a) < best_score(b);
    });

    std::vector<geometry_msgs::msg::Pose> poses{};
    for (size_t i = 0; i < std::min(sorted_msg.detections.size(), 2); i++) {
        if (not sorted_msg.detections[i].results.empty()) {
            poses.emplace_back(sorted_msg.detections[i].results[0].pose.pose);
        }
    }

    manager.update(poses);

    stamp = msg.header.stamp;
    if (num_detections != std::min(sorted_msg.detections.size(), 2)) {
        toggle_stamp = msg.header.stamp;
        tracking_id
    }

    num_detections = std::min(sorted_msg.detections.size(), 2);

    if ((rclcpp::Time(stamp) - rclcpp::Time(toggle_stamp)).nanoseconds <= 3e8
        and (rclcpp::Time(msg.header.stamp) - rclcpp::Time(toggle_stamp)).nanoseconds > 3e8
            and manager.is_initialized()) {
        node->send_command(
            rm_msgs::msg::Imu().set__roll(0.).set__pitch(bac_res.pitch).set__yaw(bac_res.yaw),
            1,
            1
        );
    } else {
        node->send_command(
            rm_msgs::msg::Imu().set__roll(0.).set__pitch(bac_res.pitch).set__yaw(bac_res.yaw),
            1,
            0 
        );
    }
}

EnergyVariableSpeedPredictor::EnergyVariableSpeedPredictor(const PredictorNodeInterface* node) {
}

PredictorNodeInterface::PredictorNodeInterface() :
    Node(kNodeName, rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),
    imu_stamped_cache(std::make_shared<message_filters::Cache<rm_msgs::msg::ImuStamped>>(4096)),
    robot_state(std::make_shared<rm_msgs::msg::State>(rm_msgs::msg::State().set__robot_id(15))),
    camera_info(std::make_shared<sensor_msgs::msg::CameraInfo>()),
    vision_follow_id(15)
{
    toggle_vision_follow_id_srv = create_service<rm_msgs::srv::ToggleVisionFollowId>(
        "/toggle_vision_follow_id", [this](
            const std::shared_ptr<rm_msgs::srv::ToggleVisionFollowId::Request> request,
            std::shared_ptr<rm_msgs::srv::ToggleVisionFollowId::Response> response
        ){
            vision_follow_id = 15;
            response->set__success(true);
    });

    toggle_vision_follow_id_cli = create_client<rm_msgs::srv::ToggleVisionFollowId>(
        "/toggle_vision_follow_id");

    set_vision_mode_srv = create_service<rm_msgs::srv::SetVisionMode>(
        "/set_vision_mode", [this](
            const std::shared_ptr<rm_msgs::srv::SetVisionMode::Request> request,
            std::shared_ptr<rm_msgs::srv::SetVisionMode::Response> response
        ){
            response->set__success(true);
            switch (request->vision_mode) {
            case 0:
                if (not predictor || dynamic_cast<EnemyPredictor*>(predictor.get()) == nullptr) {
                    predictor = std::make_unique<EnemyPredictor>(this);
                }
                break;
            case 1:
                if (not predictor || dynamic_cast<EnergyPredictor*>(predictor.get()) == nullptr) {
                    predictor = std::make_unique<EnergyPredictor>(this);
                }
                break;
            case 2:
                if (not predictor || dynamic_cast<OutpostPredictor*>(predictor.get()) == nullptr) {
                    predictor = std::make_unique<OutpostPredictor>(this);
                }
                break;
            case 3:
                if (not predictor || dynamic_cast<BasePredictor*>(predictor.get()) == nullptr) {
                    predictor = std::make_unique<BasePredictor>(this);
                }
                break;
            default:
                response->set__success(false);
                break;
            }
        }
    );

    ext_control_sub = create_subscription<rm_msgs::msg::Control>(
        "/ext_control", rclcpp::QoS(10), [this](const std::shared_ptr<rm_msgs::msg::Control> msg){
            if (dynamic_cast<PredictorInterface*>(predictor.get()) == nullptr) {
                control_pub->publish(*msg);
            }
        }
    );

    vision_sub = create_subscription<vision_msgs::msg::Detection2DArray>(
        "  /vision/tracked", rclcpp::QoS(10), [this](const std::shared_ptr<vision_msgs::msg::Detection2DArray> msg) {
            /// NOTE: Whether to keep this gate control is still up for debate.
            auto imu = get_imu(rclcpp::Time(msg->header.stamp));
            auto detections = msg->detections;

            tf2::Quaternion q;        
            q.setRPY(
                imu.roll,
                imu.pitch,
                imu.yaw
            );
            q.normalize();
            Eigen::Quaterniond transform(q.w(), q.x(), q.y(), q.z());

            for (auto& det: detections) {
                for (auto& res: det.results) {
                    Eigen::Vector3d position;
                    tf2::fromMsg(res.pose.pose, position);
                    res.pose.set__pose(
                        geometry_msgs::msg::Pose()
                        .set__position(tf2::toMsg(transform * position))
                        .set__orientation(geometry_msgs::msg::Quaternion()
                            .set__x(res.pose.pose.orientation.x + imu.roll)
                            .set__y(res.pose.pose.orientation.y + imu.pitch)
                            .set__z(res.pose.pose.orientation.z + imu.yaw)
                        )
                    );
                }
            }

            image_annotations = std::make_shared<foxglove_msgs::msg::ImageAnnotations>();
            image_annotations->set__circles(std::vector<foxglove_msgs::msg::CircleAnnotation>(
                1, foxglove_msgs::msg::CircleAnnotation()
                .set__timestamp(camera_info->header.stamp)
                .set__position(foxglove_msgs::msg::Point2()
                    .set__x(camera_info->k[2])
                    .set__y(camera_info->k[5]))
                .set__diameter(5.0)
                .set__thickness(2.5)
                .set__outline_color(foxglove_msgs::msg::Color()
                    .set__r(0.00f)
                    .set__g(1.00f)
                    .set__b(0.00f)
                    .set__a(0.85f))
                )
            );

            if (not msg->detections.empty()) {
                if (vision_follow_id == 15 && robot_state->vision_follow_enable) {
                    const double cx = camera_info->width * 0.5;
                    const double cy = camera_info->height * 0.5;

                    auto it = std::min_element(msg->detections.begin(), msg->detections.end(),
                        [&](const auto& a, const auto& b) {
                        const auto ax{a.bbox.center.position.x - cx};
                        const auto ay{a.bbox.center.position.y - cy};
                        const auto bx{b.bbox.center.position.x - cx};
                        const auto by{b.bbox.center.position.y - cy};
                        return (ax*ax + ay*ay) < (bx*bx + by*by);
                        });

                    if (not it->results.empty()) {
                        vision_follow_id = std::stoi(it->results[0].hypothesis.class_id) % 10;
                    }
                }
            }

            if (auto callback_shared{vision_callback.lock()}) {
                callback_shared->operator()(msg);
            } else {
                RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000,
                                    "Vision callback not registered.");
                return;
            }

            image_annotations_pub->publish(*image_annotations);
    });

    state_sub = create_subscription<rm_msgs::msg::State>(
        "/robot_state", rclcpp::QoS(10), [this](std::shared_ptr<rm_msgs::msg::State> state) {
            imu_stamped_cache->add(std::make_shared<rm_msgs::msg::ImuStamped>(rm_msgs::msg::ImuStamped()
                .set__header(std_msgs::msg::Header().set__stamp(get_clock()->now()))
                .set__imu(state->imu)));

            if (state->robot_id != robot_state->robot_id) {
                if (state->robot_id % 10 == 1) {
                    ballestic_solver->build(
                        get_parameter("ballestic.big.v").as_double(),
                        get_parameter("ballestic.big.k").as_double()
                    );
                } else {
                    ballestic_solver->build(
                        get_parameter("ballestic.small.v").as_double(),
                        get_parameter("ballestic.small.k").as_double()
                    );
                }
            }

            if (state->vision_follow_enable == true and robot_state->vision_follow_enable == false) {
                toggle_vision_follow_id_cli->async_send_request(
                    std::make_shared<rm_msgs::srv::ToggleVisionFollowId::Request>());
            }

            robot_state = state;
        }
    );

    camera_info_sub = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/hikcam/camera_info",
        rclcpp::QoS(10),
        [this](std::shared_ptr<sensor_msgs::msg::CameraInfo> msg){
            camera_info = msg;
        }
    );

    control_pub = create_publisher<rm_msgs::msg::Control>(
        "/control", rclcpp::QoS(10)
    );

    image_annotations_pub = create_publisher<foxglove_msgs::msg::ImageAnnotations>(
        "/vision/armor_annotations", rclcpp::QoS(10)
    );
}

rm_msgs::msg::Imu PredictorNodeInterface::get_imu(rclcpp::Time stamp) {
    if (stamp.nanoseconds() == 0) {
        stamp = get_clock()->now();
    }

    auto imu_before{imu_stamped_cache->getElemBeforeTime(stamp)};
    auto imu_after{imu_stamped_cache->getElemAfterTime(stamp)};

    if (imu_before == nullptr and imu_after == nullptr) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
            "IMU cache empty for requested stamp.");
        return rm_msgs::msg::Imu();
    } else if (imu_before == nullptr) {
        return imu_after->imu;
    } else if (imu_after == nullptr) {
        return imu_before->imu;
    }

    const rclcpp::Time t0(imu_before->header.stamp);
    const rclcpp::Time t1(imu_after->header.stamp);

    if (t1.nanoseconds() <= t0.nanoseconds()) {
        return imu_before->imu;
    }

    const double alpha = static_cast<double>(
        stamp.nanoseconds() - t0.nanoseconds())
            / static_cast<double>(t1.nanoseconds() - t0.nanoseconds());

    return rm_msgs::msg::Imu()
        .set__roll(imu_before->imu.roll + (imu_after->imu.roll - imu_before->imu.roll) * alpha)
        .set__pitch(imu_before->imu.pitch + (imu_after->imu.pitch - imu_before->imu.pitch) * alpha)
        .set__yaw(imu_before->imu.yaw + (imu_after->imu.yaw - imu_before->imu.yaw) * alpha);
}

rm_msgs::msg::State PredictorNodeInterface::get_robot_state() {
    return robot_state;
}


uint8_t PredictorNodeInterface::get_vision_follow_id() {
    return vision_follow_id;
}

BallisticResult PredictorNodeInterface::solve_ballestic(geometry_msgs::msg::Point point) {
    return ballestic_solver->query(point.x, point.y, point.z);
}

void PredictorNodeInterface::send_command(const rm_msgs::msg::Imu& imu, uint8_t gimbal_mode, bool booster_mode) {
    if (robot_state->vision_follow_enable) {
        control_pub->publish(rm_msgs::msg::Control()
            .set__imu(imu)
            .set__gimbal_mode(gimbal_mode)
            .set__booster_mode(booster_mode)
            .set__vision_follow_id(vision_follow_id)
        );
    } else {
        control_pub->publish(rm_msgs::msg::Control()
            .set__imu(robot_state->imu)
            .set__gimbal_mode(0)
            .set__booster_mode(0)
            .set__vision_follow_id(15)
        );
    }
}

void PredictorNodeInterface::annotate(const std::vector<geometry_msgs::msg::Pose>& poses) {
    static size_t color_index{0};
    static const std::array<foxglove_msgs::msg::Color, 8> COLOR{{
        foxglove_msgs::msg::Color().set__r(0.00f).set__g(1.00f).set__b(0.90f).set__a(0.85f),
        foxglove_msgs::msg::Color().set__r(1.00f).set__g(0.20f).set__b(0.60f).set__a(0.85f),
        foxglove_msgs::msg::Color().set__r(1.00f).set__g(0.85f).set__b(0.10f).set__a(0.85f),
        foxglove_msgs::msg::Color().set__r(0.20f).set__g(1.00f).set__b(0.20f).set__a(0.85f),
        foxglove_msgs::msg::Color().set__r(0.40f).set__g(0.60f).set__b(1.00f).set__a(0.85f),
        foxglove_msgs::msg::Color().set__r(1.00f).set__g(0.45f).set__b(0.10f).set__a(0.85f),
        foxglove_msgs::msg::Color().set__r(0.75f).set__g(0.35f).set__b(1.00f).set__a(0.85f),
        foxglove_msgs::msg::Color().set__r(1.00f).set__g(1.00f).set__b(1.00f).set__a(0.85f),
    }};

    for (auto pose: poses) {
        double dx = 0.0675;
        double dy = 0.0275;

        std::vector<foxglove_msgs::msg::Point2> points;
        for (double sx : {-1.0, 1.0}) {
            for (double sy : (sx > 0 ? std::vector<double>{-1.0, 1.0} : std::vector<double>{1.0, -1.0})) {
                points.emplace_back(foxglove_msgs::msg::Point2()
                    .set__x(camera_info->k[0] * (pose.position.x + sx * dx)
                        / (pose.position.y + sy * dy) + camera_info->k[2])
                    .set__y(camera_info->k[4] * (pose.position.x + sx * dx)
                        / (pose.position.y + sy * dy) + camera_info->k[5])
                );
            }
        }

        image_annotations->points.emplace_back(foxglove_msgs::msg::PointsAnnotation()
            .set__timestamp(camera_info->header.stamp)
            .set__type(foxglove_msgs::msg::PointsAnnotation::LINE_LOOP)
            .set__points(points)
            .set__outline_color(COLOR[color_index++ % COLOR.size()])
            .set__thickness(4.0) // pixels
        );
    }
}

void PredictorNodeInterface::annotate(const geometry_msgs::msg::Pose& poses) {
    annotate(std::vector<geometry_msgs::msg::Pose>(1, poses));
}

std::shared_ptr<std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray)>>
PredictorNodeInterface::register_callback(
    const std::function<void(std::shared_ptr<vision_msgs::msg::Detection2DArray)>& callback) {
    auto callback_shared {std::make_shared<std::function<
        void(std::shared_ptr<vision_msgs::msg::Detection2DArray)>>(callback)};
    vision_callback = callback_shared; // implicit conversion
    return callback_shared;
}
