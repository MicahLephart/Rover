"""Vehicle node for the collection rover."""

import rclpy

from moon_rover_bridge.rover_base import RoverBridgeNode


class CollectorRoverNode(RoverBridgeNode):
    DEFAULT_MODEL = 'collector_rover'
    # Empty = track every link in the Gazebo model. Set the `parts` parameter to a list of
    # link names (e.g. a scoop link) to track only specific rover parts.
    DEFAULT_PARTS = []

    def __init__(self):
        super().__init__('collector_rover_node')


def main(args=None):
    rclpy.init(args=args)
    node = CollectorRoverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
