#What we use to communicate with the ros2 package
import rclpy 
#Access to the nodes
from rclpy.node import Node

class CustomNode(Node):
    def __init__(self, name):
        super().__init__(name)

        self.create_timer(1, callback=self.callback)

    def callback(self):
         self.get_logger().info('hey!')




def main (args=None):
    rclpy.init(args=args)

    node = CustomNode('example_node_name')

    rclpy.spin(node)

    rclpy.shutdown()


if __name__ == "__main__":
    main()

