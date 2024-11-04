x0 = 0.7610708532037432
y0 = -0.2949797002806579
z0 = -0.507632798055196
w0 = 0.8615720352406652

x1 = 1.8324408002600754
y1 = -1.0626442975602988
z1 = 0.010095723504304578
w1 = 0.9999477680231048

x2 = 2.7513659810588527
y2 = -2.014546901801887
z2 = -0.5812638825768531
w2 = 0.8137135671779846

x3 = 1.6884347811234859
y3 = -3.204653708789601
z3 = -0.9866735201986652
w3 = 0.16270474074988645

x4 = -0.24998370412392007
y4 = -3.270619705170102
z4 = -0.9734911991709222
w4 = -0.2287189079450642

x5 = -1.1746902196971647
y5 = -2.058732448492658
z5 = -0.9999281244542478
w5 = 0.011881625227837842

x6 = -2.6731347214776107
y6 = -2.360225296683835
z6 = -0.9677342511963948
w6 = 0.2519679603497954

x7 = -3.9622917548229695
y7 = -2.321236330903385
z7 = -0.816888557251017
w7 = -0.5767933224821922

x8 = -3.616874742463826
y8 = -0.2239407907417962
z8 = -0.4111990992574577
w8 = -0.9115441600182901

x9 = -2.939343762984863
y9 = 1.48071144498626
z9 = -0.7227876523600936
w9 = -0.6910683366284002

x = [0.0,x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
y = [0.0,y0,y1,y2,y3,y4,y5,y6,y7,y8,y9]
z = [0.0,z0,z1,z2,z3,z4,z5,z6,z7,z8,z9]
w = [1.0,w0,w1,w2,w3,w4,w5,w6,w7,w8,w9]

R_theta = []

import numpy as np
from math import atan

def r_t(pose1,pose2):
    
    theta = atan((pose2[1]-pose1[1])/(pose2[0]-pose1[0])) - pose1[2]
    
    if np.isnan(theta):
        theta = 0
    
    R = ((pose2[1]-pose1[1])**2 + (pose2[0]-pose1[0])**2)**0.5
    
    return [R,theta]

for i in range(len(x)-1):
    pose_1 = [x[i],y[i],z[i]]
    pose_2 = [x[i+1],y[i+1],z[i+1]]
    data = r_t(pose_1,pose_2)
    R_theta.append(data)

array = np.array(R_theta)



import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


import math

# Set desired Euler angles (roll, pitch, yaw) in radians








def reset_turtlebot(x,y,z,w):
    rospy.wait_for_service('/gazebo/set_model_state')
    set_model_state = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    
    # Create a ModelState message and set the desired pose
    model_state = ModelState()
    

    


    model_state.model_name = 'turtlebot3_waffle'  # Replace with your TurtleBot's model name
    
    model_state.pose.position.x = x  # Set desired X position
    model_state.pose.position.y = y  # Set desired Y position
    model_state.pose.position.z = 0.0  # Set desired Z position
    model_state.pose.orientation.x = 0.0  # Set desired X orientation
    model_state.pose.orientation.y = 0.0  # Set desired Y orientation
    model_state.pose.orientation.z = z  # Set desired Z orientation
    model_state.pose.orientation.w = w  # Set desired W orientation (quaternion)
    
    # Call the service to set the new pose of the TurtleBot
    set_model_state(model_state)

    
from time import sleep

for i in range(11):
    print(i)
    reset_turtlebot(x[i],y[i],z[i],w[i])
    sleep(4)