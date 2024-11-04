#! /usr/bin/python3
# rospy for the subscriber

import rospy

# ROS Image message

from tensorflow import keras
import tensorflow_addons as tfa
# ROS Image message
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# ROS Image message -> OpenCV2 image converter
from cv_bridge import CvBridge, CvBridgeError

# OpenCV2 for saving an image
import cv2
import numpy as np
from std_msgs.msg import String

from math import sin , cos
rospy.init_node('image_subscriber')

# Instantiate CvBridge
bridge = CvBridge()
from time import sleep
from std_srvs.srv import Empty

from time import sleep

from tensorflow.keras.applications import VGG16,MobileNet
from tensorflow.keras.layers import Dropout,Dense,Flatten,Input,Activation,concatenate,Concatenate,Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
from math import atan

Goal = np.load("near_goal.npy")




pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
move = Twist()


input_shape = (160, 120, 3)
M=70
T=100
alpha=0.005#0.001
eps = 0.2

learning_rate = 0.01
weight_decay = 0.0001
batch_size = None
num_epochs = 50
image_size = 256  # We'll resize input images to this size
patch_size = 64  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 4
num_heads = 2
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 2
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier



def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


base_model = MobileNet(
    input_shape=(160,120,3),
    alpha=1.0,
    depth_multiplier=1,
    dropout=0.001,
    include_top=False,
    weights='mobilenet_1_0_224_tf_no_top.h5',#None
    input_tensor=None,

)


for layer in base_model.layers:
    layer.trainable = True    #!!!!!!!




def feature_ext():
        image_inputs = layers.Input(shape=(160,120,3)) #shape=(None, 160, 120, 3))
        goal_inputs = layers.Input(shape=(2,))
            # Augment data.

        g = Dense(32,activation = 'linear')(goal_inputs)
        g = Dense(128,activation = 'linear')(g)
        g = Dense(512,activation = 'linear')(g)
        g = Dense(2048,activation = 'linear')(g)
        g = Dense(8192,activation = 'linear')(g)
        g = Dense(19200,activation = 'relu')(g)
        print("kir to hossein")
        g= tf.reshape(g,(-1,160,120,1))
        #g=tf.expand_dims(g,axis=0)

        conCat = Concatenate()([image_inputs,g])
        inp = Conv2D(3,3, padding='same', activation="relu")(conCat)
        
        
        kir_to= base_model(inp)


        #inp=tf.expand_dims(inp,axis=0)
        

        features = Flatten()(kir_to)


        
        linear_vel = Dense(4096, activation="relu",name = 'first_lin')(features)
        linear_vel = Dense(2048, activation="relu")(linear_vel)
        linear_vel = Dense(1024, activation="relu")(linear_vel)
        linear_vel = Dense(512, activation="relu")(linear_vel)
        linear_vel = Dense(128, activation="relu")(linear_vel)
        linear_vel = Dense(32, activation="relu")(linear_vel)
        linear_vel = Dense(1,)(linear_vel)                        
        linear_net =Activation("linear", name="linear_output")(linear_vel)                  
                                                                                        
                                                                                         
        angular_vel = Dense(4096, activation="relu",name = 'first_ang')(features)
        angular_vel = Dense(2048, activation="relu")(angular_vel)
        angular_vel = Dense(1024, activation="relu")(angular_vel)
        angular_vel = Dense(512, activation="relu")(angular_vel)
        angular_vel = Dense(128, activation="relu")(angular_vel)
        angular_vel = Dense(32, activation="relu")(angular_vel)
        angular_vel = Dense(1,)(angular_vel)                        
        angular_net =Activation("linear", name="angular_output")(angular_vel)   

        
        
        
        model = Model(inputs=[image_inputs,goal_inputs], outputs=[linear_net,angular_net])

        
        return model







def scheduler(epoch, lr):
    if epoch%2 ==0:
        return lr * tf.math.exp(-0.1)
    else:
        return lr
def run_experiment(model):
    lr_planer = tf.keras.callbacks.LearningRateScheduler(scheduler, verbose=0)
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    
    losses={'linear_output':'mse','angular_output':'mse'}                      
    weight={'linear_output':1,'angular_output':4}                                                                                   
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics={ 'linear_output':"mean_squared_error",'angular_output': "mean_squared_error"},loss_weights=weight)
    
    log_dir = ''
    checkpoint_filepath = log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint( 
        #callback is an object that can perform actions at various stages of training
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    return model





freture = feature_ext()
model = run_experiment(freture)




current_pose_topic = "/odom"



def goal_publish(x,y):
    pub = rospy.Publisher('chatter', String, queue_size=10)
    #rate = rospy.Rate(1) 
    #while not rospy.is_shutdown():
    pub.publish(f'{x} {y}')
    #rate.sleep()




def relative_current_goal_pose(current,goal):
    theta = atan((goal[1]-current[1])/(goal[0]-current[0]))
    R = ((goal[1]-current[1])**2 + (goal[0]-current[0])**2)**0.5

    return [R,theta]





def r_theta_to_X_Y(r,theta):
    x = r*cos(theta)
    y = r*sin(theta)
    
    return x,y


def get_current_pose():
    current = rospy.wait_for_message(current_pose_topic, Odometry)
    data_as_list = str(current).splitlines()
    pose_current = [float(data_as_list[10][9:]),float(data_as_list[11][9:])]
    global current_pose
    current_pose = np.array(pose_current)
    #print('@@@@@@ current pose ==>  ',current_pose)
    return current_pose




def get_image():

    fucking_image = rospy.wait_for_message(image_topic, Image)
    cv2_img = bridge.imgmsg_to_cv2(fucking_image, "rgb8")
    #global image_data
    image_data = cv2.resize(cv2_img,(120,160),interpolation = cv2.INTER_AREA)
    return image_data




def contact_checker():
    data = rospy.wait_for_message("contact_checker",String)
    if eval(data.data):
        print('!!!!!!!! CONTACT !!!!!!!!')
    return eval(data.data)

def dist_reward(beta,state_d,Pre_state_d):

        return beta*(Pre_state_d - state_d) 
# Define your image topic
image_topic = "/camera/rgb/image_raw"
# Set up your subscriber and define its callback


while not rospy.is_shutdown():


    




    for episode in range(M):

            rospy.wait_for_service('/gazebo/reset_world')
            reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
            reset_world()

            # Here you should define youre environment
            # Defining the initial state 
            # including reset function


            goal = Goal[episode]  # fake goal we need a real one
            x,y = r_theta_to_X_Y(goal[0],goal[1])
            #distance_saver = deque([goal[0]], maxlen=2)
            goal_publish(x,y)
            state = get_image()
            
            for t in range(T):
                sleep(1)
                done_1 = contact_checker()
               
                current_pose = get_current_pose()
                relative_r_theta = np.array(relative_current_goal_pose(current_pose,np.array([x,y])))
                print('@@@@@@@@@@@@@@relative   ',relative_r_theta)
                #distance_saver.append(relative_r_theta[0]
                if done_1 == True:
                       

                        break
                        

                elif  relative_r_theta[0] <= eps :

                    print('******* I Got It *******')
                        
                        

                    break

                    

                else:


                        state_2 = np.expand_dims(state,axis = 0)
                        goal_2 = np.expand_dims(relative_r_theta,axis=0)

                        

                        action= model.predict([state_2,goal_2]) # take the action and goes to the naxt state : observe the next state from the env
                        #state,relative_r_theta
                        move.linear.x = action[0]
                        move.angular.z = action[1]
                        print(f'x ==> {action[0]} $ z ==> {action[1]}')
                        pub.publish(move)

                        current_pose = get_current_pose()
                        new_relative_r_theta = np.array(relative_current_goal_pose(current_pose,np.array([x,y])))

                        

                        n_state=  get_image()  # it should be the actual state  ######### what the fuck     n_state = image_data
                        
                        
                        
                                 
                                 
                        state=n_state



    rospy.signal_shutdown(' :) Break :) ')

