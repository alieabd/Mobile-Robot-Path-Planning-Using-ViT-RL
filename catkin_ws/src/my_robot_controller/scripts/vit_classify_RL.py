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
from std_srvs.srv import Empty

from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState

from time import sleep

from tensorflow.keras.layers import Dropout,Dense,Flatten,Input,Activation,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from math import atan

Goal = np.load("continue_goal_third_try.npy")
location = np.load("location_continue_third_try.npy")


num_classes_linear = 7
num_classes_angular = 34


pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
move = Twist()


input_shape = (120, 160, 3)
M=70
T=100
batch_size=10
alpha=0.005#0.001
eps = 0.2

learning_rate = 0.01
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 256  # We'll resize input images to this size
patch_size = 64  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 2
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier


data_augmentation = keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        #layers.RandomFlip("horizontal"),
        #layers.RandomRotation(factor=0.02),
        #layers.RandomZoom(
        #    height_factor=0.2, width_factor=0.2
        #),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.




def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x




#Goal MLP
#Goal is sth like np.array([[1,2]])
'''
Goal = mlp(goal,[16,64,192]) #shape=(1, 192)
tf.reshape(Goal,[6,6,3])
'''
#use it alongside image patches

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches




class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim) #Linear Projection Of Patch-Tockens
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
        #input_dim ==> encoding number 0 to num_patches -1
        #output_dim ==> self explanatary -> turns every input to output_dim
        #input_length ==> the Maximum length of input sequence 
        '''
        input_dim=10 ==> each word in input
        output_dim=4
        input_length=2
        
        >> input_data = np.array([[1,2]])
        input ->(1, 2)
        output ->[[[ 0.04502351  0.00151128  0.01764284 -0.0089057 ]
                    [-0.04007018  0.02874336  0.02772436  0.00842067]]]
        '''

        
        
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded




def create_vit_classifier():
    image_inputs = layers.Input(shape=input_shape) #shape=(None, 160, 120, 3))
    print('input shape ==> ',image_inputs.shape)
    goal_inputs = layers.Input(shape=(2,))
    # Augment data.
    augmented = data_augmentation(image_inputs) #shape=(None, 256, 256, 3)
    # Create patches.
    print('augmented ==>',augmented)
    patches = Patches(patch_size)(augmented) # shape=(None, None, 3072)
    #print(f'==> {patches.shape}')
    
    #Add Goal to Patches
    Goal = mlp(goal_inputs,[32,512,3072,12288],dropout_rate=0.05)#shape=(None, 3072)
    print('Goal shape ==> ',Goal.shape)
    #Goal = tf.reshape(Goal,[1,1,3072])
    print(f'Goal ==> {Goal.shape}') 
    print(f'patches ==> {patches.shape}') 
    patches = tf.keras.layers.Concatenate(axis=1)([tf.expand_dims(Goal, axis=1),patches]) #shape=(None, None, 3072)
    print(f'patches ==> {patches.shape}')
    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches+1, projection_dim)(patches) #shape=(1, 64, 4)
    print('encoded_patches ==> ',encoded_patches.shape)
    
    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        print("$$$im inside the for")
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention( #Self Attention mechanism
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

        
        
        
    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.2)(representation)
    #so far we created the image representation
    '''
    Note that the layers.GlobalAveragePooling1D layer could also be used 
    instead to aggregate the outputs of the Transformer block, 
    especially when the number of patches and the projection dimensions are large.
    '''
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.1)
    
    
    logits_linear = layers.Dense(num_classes_linear,activation='softmax',name='linear')(features)
    logits_angular = layers.Dense(num_classes_angular,activation='softmax',name='angular')(features)
                                                                                         
    
    
    # Create the Keras model
    model = keras.Model(inputs=[image_inputs,goal_inputs], outputs=[logits_linear,logits_angular])             
    return model



def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )
    
    losses={'linear_output':'mse','angular_output':'mse'}                     
    weight={'linear_output':1,'angular_output':4}                                                                                   ################################
    model.compile(
        optimizer=optimizer,
        loss=losses,
        metrics={ 'linear_output':"mean_squared_error",'angular_output': "mean_squared_error"},loss_weights=weight)
    
    log_dir = ''
    checkpoint_filepath = 'ep063-loss1.639-val_loss1.559.h5'
    checkpoint_callback = keras.callbacks.ModelCheckpoint( 
        #callback is an object that can perform actions at various stages of training
        checkpoint_filepath,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
    )

    model.load_weights(checkpoint_filepath)
 #   _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
  #  print(f"Test accuracy: {round(accuracy * 100, 2)}%")
   # print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")



    return model


vit_classifier = create_vit_classifier()
model = run_experiment(vit_classifier)




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
    image_data = cv2.resize(cv2_img,(160,120),interpolation = cv2.INTER_AREA)
    image_data = image_data.astype(np.int32)
    image_data = np.reshape(image_data,[120,160,3])
    return image_data.astype(np.int32)


def publish_velocity(x,z):
 
    move.linear.x = x
    move.angular.z = z
    #print(f'published velocity x=>{x} & z=>{z}')
    pub.publish(move)



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


def reset_turtlebot(x,y,z,w):
    #first lets stop the robot
    publish_velocity(0.0,0.0)
    print('*******I just stopped the robot*******')



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

velocity_ranges_1 = [-0.3, -0.2, -0.1,  0. ,  0.1,  0.2,  0.3]
velocity_ranges_2 = [-1.6, -1.5, -1.4, -1.3, -1.2, -1.1, -1. , -0.9, -0.8, -0.7, -0.6,
       -0.5, -0.4, -0.3, -0.2, -0.1, -0. ,  0.1,  0.2,  0.3,  0.4,  0.5,
        0.6,  0.7,  0.8,  0.9,  1. ,  1.1,  1.2,  1.3,  1.4,  1.5,  1.6,
        1.8]
while not rospy.is_shutdown():

    for episode in range(10):

            reset_turtlebot(location[:,episode][0],location[:,episode][1],location[:,episode][2],location[:,episode][3])
            # Here you should define youre environment
            # Defining the initial state 
            # including reset function


            goal = Goal[episode]  # fake goal we need a real one
            x,y = r_theta_to_X_Y(goal[0],goal[1])
            #distance_saver = deque([goal[0]], maxlen=2)
            goal_publish(x,y)
            
            conti = True
            trajectory = 1
            
            state = get_image()
            while conti:

                
                Collision = contact_checker()
               
                current_pose = get_current_pose()
                relative_r_theta = np.array(relative_current_goal_pose(current_pose,np.array([x,y])))
                print('@@@@@@@@@@@@@@relative   ',relative_r_theta)
                #distance_saver.append(relative_r_theta[0]
                
                
                if Collision:
                    print('#######collison#######')
                    reset_turtlebot(location[:,episode][0],location[:,episode][1],location[:,episode][2],location[:,episode][3])
                        

                elif  relative_r_theta[0] <= eps :

                    print('******* I Got It *******')
                    #reset_turtlebot(location[:,episode][0],location[:,episode][1],location[:,episode][2],location[:,episode][3])
                    conti = False    
                        

                    

                    

                else:

                        print('----------------continue----------------')
                        state_2 = np.expand_dims(state,axis = 0)
                        goal_2 = np.expand_dims(relative_r_theta,axis=0)

                        

                        predictions= model.predict([state_2,goal_2]) # take the action and goes to the naxt state : observe the next state from the env
                        #state,relative_r_theta
                        print('^^^^^^^^^^^^')
                        print(predictions)
                        print('^^^^^^^^^^^^^^^^^')
                        predicted_labels_1 = predictions[0].argmax(axis=1)  # Predicted labels for the first velocity component
                        predicted_labels_2 = predictions[1].argmax(axis=1)  # Predicted labels for the second velocity component

                        predicted_velocities_1 = velocity_ranges_1[predicted_labels_1[0]] 
                        predicted_velocities_2 = velocity_ranges_2[predicted_labels_2[0]] 

                        publish_velocity(predicted_velocities_1,predicted_velocities_2)
                        
                        print(f'x ==> {predicted_velocities_1} $ z ==> {predicted_velocities_2}')
                        

                        current_pose = get_current_pose()
                        #new_relative_r_theta = np.array(relative_current_goal_pose(current_pose,np.array([x,y])))  #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@                      

                        n_state=get_image()  # it should be the actual state  ######### what the fuck     n_state = image_dat
                                                                 
                        state=n_state



    rospy.signal_shutdown(' :) Break :) ')

