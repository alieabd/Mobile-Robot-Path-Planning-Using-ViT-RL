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
from geometry_msgs.msg import PoseStamped
from move_base_msgs.msg import MoveBaseActionResult
from nav_msgs.msg import Odometry


import numpy as np



image_data = np.load('test_image_data.npy')
#image_data_per_episode = np.load("image_data_per_episode.npy")
velocity_data = np.load("test_velocity.npy")
#velocity_data_per_episode = np.load("velocity_data_per_episode.npy")
current_pose_data = np.load('test_current_pose.npy')
#current_pose_data_per_episode = np.load('current_pose_per_episode.npy')
episode_length = np.load("test_episode.npy")




rospy.init_node('image_listener')
rate = rospy.Rate(0.5)




# Instantiate CvBridge
bridge = CvBridge()






def goal_subscriber_callback(goal):
    print("************************* I Got It!!! *************************")
    #print(msg)
    global episode_length
    episode_length = np.concatenate((episode_length,np.array([image_data.shape[0]])),axis=0)
    print(f'this was your {episode_length.shape}Th episode')
    print(f'lenght of this episode ==> {episode_length[-1]}')







def get_current_pose(current):
    data_as_list = str(current).splitlines()
    pose_current = [float(data_as_list[10][9:]),float(data_as_list[11][9:]),float(data_as_list[16][9:])]
    current_pose = np.array(pose_current)
    return current_pose



def callback(image,vel,current):
    
    #process image
    try:
        # Convert your ROS Image message to OpenCV2
        cv2_img = bridge.imgmsg_to_cv2(image, "rgb8")
        cv2_img = cv2.resize(cv2_img,(160,120),interpolation = cv2.INTER_AREA)
    except CvBridgeError :
        print("$$$$$Erros that you deleted")
    global image_data
    image_data = np.concatenate((image_data,np.reshape(cv2_img,[1,160,120,3])),axis=0)
    print(f'image ==> {image_data.shape}')






    #process vel
    data_as_list = str(vel).splitlines()
    vel_orientation_current = [[float(data_as_list[1][5:]),float(data_as_list[2][5:]),float(data_as_list[3][5:])],
                        [float(data_as_list[5][5:]),float(data_as_list[6][5:]),float(data_as_list[7][5:])]]
    #print(vel_orientation_current)
    global velocity_data
    velocity_data = np.concatenate((velocity_data,np.reshape(vel_orientation_current,[1,2,3])))
    print(f'velocity ==> {velocity_data.shape}')




    #process current pose
    current_pose = get_current_pose(current)
    global current_pose_data
    current_pose_data = np.concatenate((current_pose_data,np.reshape(current_pose,[1,1,3])))
    print(f'current pose ==> {current_pose_data.shape}')




goal_pose_topic = "/move_base_simple/goal"
goal_sub = rospy.Subscriber(goal_pose_topic,PoseStamped,goal_subscriber_callback)











# Define your image topic
image_topic = "/camera/rgb/image_raw"
velocity_topic = "/cmd_vel"
current_pose_topic = "/odom"
# Set up your subscriber and define its callback
image_sub = message_filters.Subscriber(image_topic, Image)
vel_sub = message_filters.Subscriber(velocity_topic, Twist)
current_sub = message_filters.Subscriber(current_pose_topic,Odometry)



ts = message_filters.ApproximateTimeSynchronizer([image_sub,vel_sub,current_sub],10,1,allow_headerless=True)    
ts.registerCallback(callback)    



rospy.spin()


inp = input('do you want to save it? ')
if inp =="y":
    #save image data per episode saved in the axis=0 at the end of episode
    #image_data_per_episode = np.concatenate((image_data_per_episode,image_data),axis=0)
    np.save("test_image_data",image_data)


    #save velocity data per episode saved in the axis=0 at the end of episode
    #velocity_data_per_episode = np.concatenate((velocity_data_per_episode,velocity_data),axis=0)
    np.save("test_velocity",velocity_data)


    #save pose and orientation of current data
    #current_pose_data_per_episode = np.concatenate((current_pose_data_per_episode,current_pose_data),axis=0)
    np.save('test_current_pose',current_pose_data)


    #save length of each episode
    np.save("episode_length",episode_length)



 

print("The Episode Ended!")