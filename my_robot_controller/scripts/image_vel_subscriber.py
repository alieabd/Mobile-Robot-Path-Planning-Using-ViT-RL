#! /usr/bin/python3
# rospy for the subscriber
import rospy
# ROS Image message
from sensor_msgs.msg import Image
# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError
# OpenCV2 for saving an image
import cv2
from geometry_msgs.msg import Twist
import message_filters


import numpy as np



image_data = np.load('image_data.npy')
image_data_per_episode = np.load("image_data_per_episode.npy")
velocity_data = np.load("velocity_data.npy")


rospy.init_node('image_listener')
rate = rospy.Rate(0.5)


# Instantiate CvBridge
bridge = CvBridge()

velocity_topic = "/cmd_vel"



def callback(image,vel):

    #process image
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(image, "rgb8")
    except CvBridgeError :
        print("$$$$$Erros that you deleted")
    global image_data
    image_data = np.concatenate((image_data,np.reshape(cv2_img,[1,1080,1920,3])),axis=0)
    print(f'image ==> {image_data.shape}')

    # Save your OpenCV2 image as a jpeg 
    #print('Image Saved')
    #cv2.imwrite('~/project/images/camera_image.jpeg', cv2_img)


    #process vel
    data_as_list = str(vel).splitlines()
    vel_orientation_current = [[float(data_as_list[1][5:]),float(data_as_list[2][5:]),float(data_as_list[3][5:])],
                        [float(data_as_list[5][5:]),float(data_as_list[6][5:]),float(data_as_list[7][5:])]]
    #print(vel_orientation_current)
    global velocity_data
    velocity_data = np.concatenate((velocity_data,np.reshape(vel_orientation_current,[1,2,3])))
    print(f'velocity ==> {velocity_data.shape}')



# Define your image topic
image_topic = "/camera/rgb/image_raw"
velocity_topic = "/cmd_vel"
# Set up your subscriber and define its callback
image_sub = message_filters.Subscriber(image_topic, Image)
vel_sub = message_filters.Subscriber(velocity_topic, Twist)

ts = message_filters.ApproximateTimeSynchronizer([image_sub,vel_sub],10,1,allow_headerless=True)    
ts.registerCallback(callback)    


rospy.spin()


#save image data per episode saved in the axis=0 at the end of episode
image_data_per_episode = np.concatenate((image_data_per_episode,image_data),axis=0)
np.save("image_data_per_episode",image_data_per_episode)
np.save("velocity_data",velocity_data)

print("The Episode Ended!")


