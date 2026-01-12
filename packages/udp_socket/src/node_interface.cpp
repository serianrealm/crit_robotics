#include "udp_socket/node_interface.hpp"

#include <rclcpp/logging.hpp>
#include <rclcpp/timer.hpp>
#include <rclcpp/utilities.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/generic_subscription.hpp>
#include <rclcpp/serialized_message.hpp>

#include <boost/asio.hpp>

#include <chrono>
#include <exception>
#include <stop_token>
#include <utility>

#include <rosidl_typesupport_introspection_cpp/message_introspection.hpp>
#include <rosidl_typesupport_cpp/message_type_support.hpp>

#include <yaml-cpp/yaml.h>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <resource_retriever/retriever.hpp>

const char* UdpSocketNodeInterface::node_name = "udp_socket";

rclcpp::NodeOptions UdpSocketNodeInterface::options = rclcpp::NodeOptions()
    .use_intra_process_comms(true)
    .automatically_declare_parameters_from_overrides(true);

UdpSocketNodeInterface::UdpSocketNodeInterface(): 
    rclcpp::Node(node_name, options),
    logger(get_logger()),
    ctx(),
    socket(ctx),
    work_guard(asio::make_work_guard(ctx)) 
{
    auto packet_size = get_parameter_or<int>("packet_size", 1024);
    buffer = std::make_shared<std::string>(packet_size, '\0');

    auto ip_addr = get_parameter_or<std::string>("ip_addr", "127.0.0.1");
    auto network = boost::asio::ip::make_network_v4(ip_addr + "/24");

    auto local_port = get_parameter_or("local_port", 6000);
    socket.open(asio::ip::udp::v4());
    socket.set_option(asio::socket_base::broadcast(true));
    socket.bind(asio::ip::udp::endpoint(asio::ip::udp::v4(), local_port));

    auto remote_port = get_parameter_or("remote_port", 6000);
    endpoint = asio::ip::udp::endpoint(network.broadcast(), remote_port);

    auto timeout = static_cast<std::chrono::duration<double>>(get_parameter_or<double>("timeout", 1.0));

    auto daemon = [this]() -> void {
        if (executor.joinable()) {
            return;
        }
        async_activate();
        executor = std::jthread (
            [this](std::stop_token st) {
                try {
                    ctx.run();
                } catch (const std::exception& ex) {
                    RCLCPP_ERROR_STREAM(logger, "Network thread error: " << ex.what());
                }
            });
    };

    timer = create_timer(timeout, daemon);
}

UdpSocketNodeInterface::~UdpSocketNodeInterface() {
    ctx.stop();
}

void UdpSocketNodeInterface::transfer_message(const std::shared_ptr<std::vector<unsigned char>>& data) {
    socket.async_send_to(
        asio::buffer(*data),
        endpoint,
        [this](std::error_code ecc, std::size_t bytes_sent) {
            RCLCPP_INFO(logger, "I sent %zu bytes", bytes_sent);
        });
}

void UdpSocketNodeInterface::async_activate() {
    socket.async_receive_from(
        asio::buffer(*buffer),
        endpoint,
        [this](std::error_code ecc, std::size_t bytes) {
            if (!ecc) {
                if (callback && *callback) {
                    (*callback)(buffer);
                } else {
                    RCLCPP_ERROR(logger, "Callback must be set before event loop.");
                }
            }
            async_activate();
        });
}

GenericUdpSocket::GenericUdpSocket() {
    auto qos =  rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_t{
        RMW_QOS_POLICY_HISTORY_KEEP_LAST,
        10UL,
        RMW_QOS_POLICY_RELIABILITY_RELIABLE,
        RMW_QOS_POLICY_DURABILITY_VOLATILE,
        {0UL, 0UL},
        {0UL, 0UL},
        RMW_QOS_POLICY_LIVELINESS_SYSTEM_DEFAULT,
        {0UL, 0UL},
        false
    }));

    auto url = get_parameter_or("url", std::string("package://config/metadata/default.yaml"));

    resource_retriever::Retriever retriever;

    auto res = retriever.get(url);
    auto metadata = YAML::Load(std::string(
        reinterpret_cast<const char*>(res.data.get()),
        res.size
    ));

    for (auto sub : metadata["subscriptions"]) {
        auto topic = sub["topic"].as<std::string>();
        auto type = sub["type"].as<std::string>();

        create_generic_subscription(
            topic,
            type,
            qos,
            [this](std::shared_ptr<rclcpp::SerializedMessage> msg){
                auto serialized_message = msg->get_rcl_serialized_message();
                auto buffer = std::vector<unsigned char>(serialized_message.buffer_length);
                auto ts = get_message_typesupport_handle();
                rmw_deserialize(&serialized_message, type_support);
                auto data = std::make_shared<std::vector<unsigned char>>(
                    serialized_message.buffer,
                    serialized_message.buffer + serialized_message.buffer_length
                );
                transfer_message(data);
            }
        );
    }

}
