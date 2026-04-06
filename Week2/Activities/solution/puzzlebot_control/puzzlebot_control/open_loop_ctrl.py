
# Imports
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float32
from geometry_msgs.msg import Twist


#Class Definition
class OpenLoopCtrl(Node):
    def __init__(self):
        super().__init__('open_loop_ctrl')

        self.wait_for_ros_time()

        # Publisher to /cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        # Time-based control variables
        self.state = 0  # 0: forward, 1: rotate, 2: backward, 3: stop
        self.state_start_time = self.get_clock().now()

        # Define speeds
        self.linear_speed = 0.2  # m/s
        self.angular_speed = 0.5  # rad/s

        # Define durations (seconds)
        self.forward_time = 2.0 / self.linear_speed   # Time to move 2m
        self.rotate_time = 3.1416 / self.angular_speed  # Time to rotate 180 deg
        self.backward_time = self.forward_time

        # Timer to update state machine
        self.timer_period = 0.2  # 10 Hz control loop
        self.timer = self.create_timer(self.timer_period, self.control_loop)

        self.get_logger().info('Open loop controller initialized!')
        
    def control_loop(self):
        now = self.get_clock().now()
        elapsed_time = (now - self.state_start_time).nanoseconds * 1e-9

        self.get_logger().info(f"Start: {self.state_start_time.nanoseconds * 1e-9}, NOW: {now.nanoseconds * 1e-9:.2f}s")
        self.get_logger().info(f"State: {self.state}, Elapsed: {elapsed_time:.2f}s")

        cmd = Twist()

        if self.state == 0:
            # Move forward
            cmd.linear.x = self.linear_speed
            self.get_logger().info('Moving forward...')
            if elapsed_time >= self.forward_time:
                self.state = 1
                self.state_start_time = now
                self.get_logger().info('Finished moving forward. Starting rotation...')

        elif self.state == 1:
            # Rotate 180 degrees
            cmd.angular.z = self.angular_speed
            self.get_logger().info('Rotating 180 degrees...')
            if elapsed_time >= self.rotate_time:
                self.state = 2
                self.state_start_time = now
                self.get_logger().info('Finished rotation. Moving backward...')

        elif self.state == 2:
            # Move backward (back to starting position)
            cmd.linear.x = self.linear_speed
            self.get_logger().info('Moving back...')
            if elapsed_time >= self.backward_time:
                self.state = 3
                self.state_start_time = now
                self.get_logger().info('Finished moving back. Stopping...')

        elif self.state == 3:
            # Stop
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            self.get_logger().info('Stopped.')
            # Optionally: cancel the timer after stopping
            self.timer.cancel()

        # Publish velocity command
        self.cmd_vel_pub.publish(cmd)


    # Wrap to Pi function
    def wrap_to_Pi(self,theta):
        result = np.fmod((theta+np.pi),(2*np.pi))
        if (result<0):
            result += 2 * np.pi
        return result - np.pi

    def wait_for_ros_time(self):
        self.get_logger().info('Waiting for ROS time to become active...')
        while rclpy.ok():
            now = self.get_clock().now()
            if now.nanoseconds > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info(f'ROS time is active! Start time: {now.nanoseconds * 1e-9:.2f}s')

#Main
def main(args=None):
    rclpy.init(args=args)

    node = OpenLoopCtrl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():  # Ensure shutdown is only called once
            rclpy.shutdown()
        node.destroy_node()

#Execute Node
if __name__ == '__main__':
    main()
