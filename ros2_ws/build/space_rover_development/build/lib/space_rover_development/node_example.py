#What we use to communicate with the ros2 package
import rclpy 
#Access to the nodes
from rclpy.node import Node

def main (args=Node):
    rclpy.init(args=args)

    node = Node('example_node_name')

    node.get_logger().info("hey!")

    rclpy.shutdown()


if __name__ == "__main__":
    main()

