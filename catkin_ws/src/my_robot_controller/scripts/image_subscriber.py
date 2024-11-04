#! /usr/bin/python3
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
import numpy as np
import time

import rospy
rospy.init_node('image_subscriber')


# Instantiate CvBridge
bridge = CvBridge()



def callback(image):

    #process image
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(image, "rgb8")
        global image_data
        image_data = cv2.resize(cv2_img,(160,120),interpolation = cv2.INTER_AREA)
    except CvBridgeError :
        print("$$$$$Erros that you deleted")
    
    print(f'image ==> {image_data.shape}')



# Define your image topic
image_topic = "/camera/rgb/image_raw"
# Set up your subscriber and define its callback

while not rospy.is_shutdown():
    rospy.Subscriber(image_topic, Image, callback) 
    print("kir to hossein")

