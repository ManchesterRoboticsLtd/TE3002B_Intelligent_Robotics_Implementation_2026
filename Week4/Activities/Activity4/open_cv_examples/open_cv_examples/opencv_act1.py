import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge
import numpy as np

class ColorDetectionNode(Node):

    def __init__(self):
        super().__init__('color_detection_node')



def main(args=None):
    rclpy.init(args=args)
    color_detection_node = ColorDetectionNode()

    try:
        rclpy.spin(color_detection_node)
    except KeyboardInterrupt:
        pass
    finally:
        color_detection_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()