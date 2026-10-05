'''
- rclpy is the ROS2 Client Library for Python that is included with any standard installation of ROS2 distributions.
- rclpy provides the standard Application Programming Interface (API)
'''

import rclpy

# Node class is the main class of rclpy that implements the basic ROS2 nodes functionality.
from rclpy.node import Node

'''
- geometry_msgs.msg.Twist is the type of the message that this code sends
- geometry_msgs.msg.Twist is a class containing components
of the linear and angular velocity vectors that are used to control the 2nd mobile robot.

- This message is sent throught the topic: /model/vechile_blue/cmd_vel.
Make sure that this is the correct topic and modify the code if it is not.
'''
from geometry_msgs.msg import Twist

'''
- The sensor information is transferred as an odometry message
- The odometry message contains information about position, orientation and velocities of the robot.
'''
from nav_msgs.msg import Odometry

# we use this function to convert the orientation expressed as a quaternion
# into an angle
# namley, the orientation in the odometry message is expressed as a quaternion
# and we need to convert it to the rotation angle
from tf_transformations import euler_from_quaternion

# here we are controlling the blue robot
# change this if you want to control the other robot
# this topic is used to send Twist control commands
topic1 = '/model/vehicle_blue/cmd_vel'
# this is used to recieve Pose sensor measurements
topic2 = '/model/vehicle_blue/odometry'

# we need to track time
import time
# for math functions
import math

# this is the class that implements the controller
# the class is inherited from the Node class
class ControllerNode(Node):
    #constructor
    #(xdu, ydu) - coordinates of the desired target point
    # Kvu, Kthetau - control parameters for the linear and angular velocities
    def __init__(self,xdu,ydu,Kvu,Kthetau):
        # call the constructor of the Node class
        # and give the name of the node
        super().__init__('controller_node')

        # (xdu,ydu) - desired point
        self.xd=xdu
        self.yd=ydu
        # Kv- control parameter for linear velocity
        self.Kv=Kvu
        # Ktheta- control parameter for angular velocity
        self.Ktheta=Kthetau

        # this is the pose that is continiously being updated
        # on the basis of the message recieved from the robot
        self.lastRecievedMsg = Odometry()
        # measure the intial time
        self.intialTime = time.time()
        # this is the time stamp when self.lastRecievedMsg is updated
        self.lastRecievedMsgTime=time.time()
        # this is the control message we are sending
        self.controlVel=Twist()
        # set the control velocity to zero at the beginning

        # componentns of the linear velocity vector
        self.controlVel.linear.x=0.0
        self.controlVel.linear.y=0.0
        self.controlVel.linear.z=0.0

        # components of the angular velocity vector
        # note that since the robot is in a 2D space only the z componnent of the velocity vector
        # matters, and this is angular velcotiy around the axis perpendicular to the computer screen
        self.controlVel.angular.x=0.0
        self.controlVel.angular.y=0.0
        self.controlVel.angular.z=0.0

        # control publisher
        self.ControlPublisher = self.create_publisher(Twist,
                                                     topic1,
                                                     10)

        self.PoseSubscriber = self.create_subscription(Odometry,
                                                       topic2,
                                                       self.SensorCallback,
                                                       10)
        # how often the message are being sent
        self.period=0.250
        # timer object

        # timer object
        # ControlFunction is the function for calculating and sending control actions
        self.timer = self.create_timer(self.period, self.ControlFunction)

    #function for recieving sensor information
    def SensorCallback(self,receivedMsg):
        self.lastRecievedMsg = receivedMsg

        #record time of the received message
        self.lastRecievedMsgTime = time.time()
    
    # function for calculating and sending control actions
    def ControlFunction(self):

        # extract the current position and orientation
        xc = self.lastRecievedMsg.pose.pose.position.x
        yc = self.lastRecievedMsg.pose.pose.position.y

        # extract the quaternion from the message
        quat = self.lastRecievedMsg.pose.pose.orientation
        # list with quaternion components
        quatl = [quat.x, quat.y, quat.z, quat.w]

        # extract the angle, only the yaw angle is changing
        (roll, pitch, yaw) = euler_from_quaternion(quatl)

        # this is the orientation angle
        thetac = yaw

        # calculate the desired orientation
        thetaD = math.atan2((self.yd-yc), (self.xd-xc))

        # calculate the desired orientation
        thetaD = math.atan2((self.yd-yc), (self.xd-xc))

        xvel = self.Kv*math.sqrt((self.xd-xc)**2+(self.yd-yc)**2)
        thetavel = self.Ktheta*(thetaD-thetac)

        # This is the twist Message that we are sending
        # here, we are creating a control velocity actions/command we want to send
        self.controlVel.linear.x=xvel
        self.controlVel.linear.y=0.0
        self.controlVel.linear.z=0.0
        self.controlVel.angular.x=0.0
        self.controlVel.angular.y=0.0
        self.controlVel.angular.z=thetavel

        print("Sending the control command")
        # publish the control message
        self.ControlPublisher.publish(self.controlVel)

        # print the sensor information on the screen for logging
        print("Recieved pose:")
        timeDiff=self.lastRecievedMsgTime

        print(f'Time,x,y,theta:({timeDiff:3f},{xc:3f},{yc:3f},{thetac:3f})')




def main(args=None):
    # before using rclpy, we need to call the init function
    rclpy.init(args=args)
    # select the control parameters
    Kv_u=0.5
    Ktheta_u=0.5
    # set the coordinates of the desired point
    xd_u = 15
    yd_u = 15

    # create the node
    TestNode = ControllerNode(xd_u,yd_u,Kv_u,Ktheta_u)
    # start the node
    rclpy.spin(TestNode)
    # destroy the node
    TestNode.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
