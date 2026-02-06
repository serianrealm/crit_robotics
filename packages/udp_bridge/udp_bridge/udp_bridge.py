import rclpy

from .node_interface import UdpBridge

def main():
    rclpy.init()

    node = UdpBridge()

    rclpy.spin(node)

    rclpy.shutdown()

if __name__ == "__main__":
    main()

