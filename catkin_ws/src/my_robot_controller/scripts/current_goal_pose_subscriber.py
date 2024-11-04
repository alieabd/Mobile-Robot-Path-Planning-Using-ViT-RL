#! /usr/bin/python3
# rospy for the subscriber
import rospy
import message_filters
import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import PoseStamped

rospy.init_node('current_goal_subscriber')
rospy.loginfo("node started")

pose_orien_data = np.load("pose_orien_data.npy")

def pose_callback(current,goal):
    print(current)
    print(goal)
    print("Received Pose!")


    data_as_list = str(current).splitlines()
    pose_orientation_current = [[float(data_as_list[9][9:]),float(data_as_list[10][9:]),float(data_as_list[11][9:])],
                        [float(data_as_list[13][9:]),float(data_as_list[14][9:]),float(data_as_list[15][9:])]]
    #print(pose_orientation_current)



    data_as_list = str(goal).splitlines()
    pose_orientation_goal = [[float(data_as_list[8][7:]),float(data_as_list[9][7:]),float(data_as_list[10][7:])],
                        [float(data_as_list[12][7:]),float(data_as_list[13][7:]),float(data_as_list[14][7:])]]
    #print(pose_orientation_goal)

    pose = np.array([pose_orientation_current,pose_orientation_goal])
    print(f"pose shape ==> {pose.shape}")
    global pose_orien_data
    print(f'pose_orien_data ==> {pose_orien_data.shape}')
    pose_orien_data = np.concatenate((pose_orien_data,np.reshape(pose,[1,2,2,3])))
    #print(f"pose ==> {pose_orien_data.shape}")


    np.save("pose_orien_data",pose_orien_data)
    print(f"data saved ==> {pose_orien_data.shape}")


goal_pose_topic = "/move_base_simple/goal"
current_pose_topic = "/initialpose"
# Set up your subscriber and define its callback
#TimeSynchronizer filter synchronizes incoming channels and outputs them in the form of a single callback
current_pose_sub = message_filters.Subscriber(current_pose_topic, PoseWithCovarianceStamped) 
goal_pose_sub = message_filters.Subscriber(goal_pose_topic, PoseStamped)
ts = message_filters.ApproximateTimeSynchronizer([current_pose_sub,goal_pose_sub],10,10)    
ts.registerCallback(pose_callback)

rospy.spin()


