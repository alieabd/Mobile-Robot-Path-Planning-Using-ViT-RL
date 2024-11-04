#! /usr/bin/python3
# rospy for the subscriber
import rospy
from time import sleep

from geometry_msgs.msg import Twist

rospy.init_node('velocity_subscriber')
rate = rospy.Rate(1)


def velocity_callback(msg):
    print("Received velocity!")    
    print(msg)
    #rate.sleep()
    sleep(1)



velocity_topic = "/cmd_vel"
# Set up your subscriber and define its callback
while not rospy.is_shutdown():
    rospy.Subscriber(velocity_topic, Twist, velocity_callback) 
    
