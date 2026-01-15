"""Entry point for launching the OpenVINO tracking pipeline."""

import rclpy

from .node import SequentialTracker

def main():
    rclpy.init()

    node = SequentialTracker()

    rclpy.spin(node)
    
    rclpy.shutdown()

if __name__ == "__main__":
    main()
