#! /usr/bin/python3
# rospy for the subscriber

import rospy

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


from tensorflow.keras.layers import Dropout,Dense,Flatten,Input,Activation,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque
from math import atan

Goal = np.load("near_goal.npy")




pub = rospy.Publisher('/cmd_vel',Twist,queue_size=1)
move = Twist()




M=70
T=100
batch_size=10
alpha=0.005#0.001
eps = 0.2
input_shape = (160, 120, 3)
learning_rate = 0.005#used to be 0.01
weight_decay = 0.0001
Batch_size = 256
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
mlp_head_units = [2048, 1024]



data_augmentation = tf.keras.Sequential(
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




def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x




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
    
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
    

def Double_Q_network():
    state= Input(shape=(1024))
    action=Input(shape=(2,))
    
    x_1 = Dense(512,activation = "relu")(state)
    x_1 = Dense(256, activation="relu")(x_1)
    x_1=Dense(128, activation="relu")(x_1)
    x_th=Dense(64, activation="relu")(x_1)
   
    x_2 = Dense(4, activation="relu")(action)#used to be 20
    x_2 = Dense(16, activation="relu")(x_2)#used to be 50
    x_vel=Dense(30, activation="relu")(x_2)
 #   x_vel=Dense(64, activation="relu")(x_2)

    

    mixed = concatenate([x_vel,x_th])#???????
    x = Dense(32, activation="relu")(mixed)
    x = Dense(16,activation = 'relu')(x)
    out = Dense(1,activation = 'linear')(x)

    
    Q_1 = Model(inputs = [state,action],outputs = out,name='first_Q')
    Q_1.compile(optimizer=Adam(learning_rate = alpha),loss = 'mse') 
    
###################################################################################################    

    y_1 = Dense(512, activation="relu")(state)
    y_1 = Dense(256, activation="relu")(y_1)
    y_1=Dense(128, activation="relu")(y_1)
    y_th=Dense(64, activation="relu")(y_1)
    
    y_2 = Dense(4, activation="relu")(action)#used to be 20
    y_2 = Dense(16, activation="relu")(y_2)#used to be 50
    y_vel = Dense(30, activation="relu")(y_2)
    #y_vel=Dense(64, activation="relu")(y_2)
    
    
    y=concatenate([y_vel,y_th])#?????
    
    y = Dense(32, activation="relu")(y)
    y = Dense(16, activation="relu")(y)
    out_2=Dense(1, activation="relu")(y)
    

    
    Q_2 = Model(inputs = [state,action],outputs = out_2,name='second_Q')
    
    Q_2.compile(optimizer=Adam(learning_rate = alpha), loss = 'mse') #learning_rate = alpha ?????????
    
    return Q_1,Q_2
    



def feature_ext():
    image_inputs = Input(shape=input_shape) #shape=(None, 32, 32, 3)
    goal_inputs = Input(shape=(2,))
    # Augment data.
    augmented = data_augmentation(image_inputs) #shape=(None, 72, 72, 3)
    # Create patches.
    patches = Patches(patch_size)(augmented) #shape=(None, None, 108)
    
    #Add Goal to Patches
    Goal = mlp(goal_inputs,[32,512,3072,12288],dropout_rate=0.05) #shape=(1, 192)  ##############

    patches = tf.keras.layers.Concatenate(axis=1)([tf.expand_dims(Goal, axis=1),patches])    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches+1, projection_dim)(patches) #shape=(None, 144, 64)
    
    
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
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
        representation = layers.Dropout(0.3)(representation)
        #so far we created the image representation
    
        # Add MLP.
        features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
        
    model = Model(inputs=[image_inputs,goal_inputs], outputs=[features])             
    return model




def Actor(state_Dims):
    features = layers.Input(shape=(1024,))
    
    linear_vel = layers.Dense(512, activation="relu" , name = 'first_lin')(features)
    linear_vel = layers.Dense(128, activation="relu")(linear_vel)
    linear_vel = layers.Dense(32, activation="relu")(linear_vel)
    linear_vel = layers.Dense(1,)(linear_vel)                        
    linear_net =Activation("linear", name="linear_output")(linear_vel)                  
     
     
                                                                                         
    angular_vel = layers.Dense(512, activation="relu",name = 'first_ang')(features)
    angular_vel = layers.Dense(128, activation="relu")(angular_vel)
    angular_vel = layers.Dense(32, activation="relu")(angular_vel)
    angular_vel = layers.Dense(1,)(angular_vel)                        
    angular_net =Activation("linear", name="angular_output")(angular_vel)
    
    model = Model(inputs = features , outputs = [linear_net,angular_net])
    
    return model



def value_net(state_dim,Neurons=1024):
    indim_1=Input(shape=(state_dim,))
    
    x=Dense(Neurons,activation="relu")(indim_1)
    x=Dense(512,activation="relu")(x)
    x=Dense(128,activation="relu")(x)
    x=Dense(32,activation="relu")(x)
    x=Dense(1,activation="relu")(x)
    
    value=Activation("linear",name="val_out")(x)
    
    value_net=Model(inputs=indim_1 , outputs=value , name="val_net")
    
    value_net.compile(optimizer=Adam(learning_rate = alpha), loss='mse')
    
    return value_net





def target_value_net(state_dim,Neurons=1024):
    indim_2=Input(shape=(state_dim,))
    
    x=Dense(Neurons,activation="relu")(indim_2)
    x=Dense(512,activation="relu")(x)
    x=Dense(128,activation="relu")(x)
    x=Dense(32,activation="relu")(x)
    x=Dense(1,activation="relu")(x)
    
    value=Activation("linear",name="val_out")(x)
    
    target_value_net=Model(inputs=indim_2 , outputs=value , name="t_val_net")
   
    
    return target_value_net




def trainable_reward_func(in_Dims):
    features= Input(shape=(in_Dims,))
    
    linear_vel = Dense(512, activation="relu" , name = 'first_lin')(features)
    linear_vel = Dense(128, activation="relu")(linear_vel)
    linear_vel = Dense(32, activation="relu")(linear_vel)
    linear_vel = Dense(1,)(linear_vel)                        
    linear_net =Activation("linear", name="linear_output")(linear_vel)                  
     
     
                                                                                         
    angular_vel = Dense(512, activation="relu",name = 'first_ang')(features)
    angular_vel = Dense(128, activation="relu")(angular_vel)
    angular_vel = Dense(32, activation="relu")(angular_vel)
    angular_vel = Dense(1,)(angular_vel)                        
    angular_net =Activation("linear", name="angular_output")(angular_vel)
    
    ConCat = concatenate([linear_net , angular_net] , axis = -1)
    
    x=Dense(32,activation="relu")(ConCat)
    x=Dense(16,activation="relu")(x)
    x=Dense(1)(x)
    
    out=Activation("linear",name="rew_out")(x)
    
    
    reward_net = Model(inputs=features,outputs=out,name="TPr_net")
    
    return reward_net




class SoftActorCritic:

    def __init__(self, action_dim,state_dim, epoch_step=1, learning_rate=0.009):#used to be 0.03
        self.state = Input(shape = (160,120,3)) 
        self.goal =  Input(shape = (2,))
        self.Actor = Actor(state_dim)
        self.q_net = Double_Q_network()  
        self.v_net = value_net(state_dim,1024)
        self.target_v_net = target_value_net(state_dim,1024)
        self.backBone = feature_ext()
        self.reward_net = trainable_reward_func(state_dim) 
        #self.log_dir = 'project/proj/'
        
        self.backBone.load_weights('Back_bone.h5')
        self.backBone.trainable = False
        #self.Actor.load_weights('actor.h5')
        self.reward_net.load_weights('reward.h5') 
        self.model  = self.put_net_together()
        
        self.memory = deque([], maxlen=2500)
        self.epoch = epoch_step
        
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate)
        #self.reward_optimizer = tf.keras.optimizers.Adam(learning_rate)


    def sample_action(self,state,goal):
        st = self.backBone([state,goal])
        th,vel=self.Actor(st)
        return th,vel
    
    def random_goal(self):
        return np.random.rand(2)
    
    def sample_state(self):
        return np.random.rand(160,120,3)*255


    def update_D_Qn(self,n_state,action,Lim,G_reward,C_reward,traj_reward,state,d_reward):
        n_st_value = self.target_v_net(n_state)

        r = self.reward_net(state) ###############################3
        
        const = 100.0 #used to be 10.0
        if abs(tf.reduce_mean(r)) <= Lim:
            
                target_Q = d_reward + traj_reward + G_reward + C_reward + const*r + n_st_value
        else:
                target_Q = d_reward + traj_reward + G_reward + C_reward + const*r + n_st_value
        
        #print(f'!!!!!!!!!!!!!!fucking D Reward ==> {d_reward}')
        #print(f'$$$$$$$$$$$$$$ reward summation ==> {d_reward + traj_reward + G_reward + C_reward + const*r}')
        
        print('target_Q ====' ,target_Q)
        self.q_net[0].fit(x=[n_state,action], y = target_Q, epochs = self.epoch,verbose=1)
        self.q_net[1].fit(x=[n_state,action], y= target_Q, epochs = self.epoch,verbose=1)
        #print('EOF')

    def update_value_net(self,st,act,batch_size):
            f_Q = tf.stop_gradient(self.q_net[0].predict([st,act]))
            sec_Q = tf.stop_gradient(self.q_net[1].predict([st,act]))




            Q = tf.minimum(f_Q,sec_Q)

            
            act_th = act[0:batch_size,0].reshape(batch_size,1)
            act_vel = act[0:batch_size,1].reshape(batch_size,1)
            
            act_th = 0.1*tf.math.log(abs(act_th) + 0.00001)
            act_vel = 0.1*tf.math.log(abs(act_vel) + 0.00001)
            
            bel_back = Q - act_th - act_vel

            self.v_net.fit(st, bel_back,epochs=1, verbose=1)
           

    def updateTargetModel(self,tau=0.3):######################
        weights= self.v_net.get_weights()
        target_net = self.target_v_net.get_weights()
        
        for i in range(len(weights)):

            new_weight = tau*weights[i]+ (1-tau)*target_net[i]
            target_net[i] = new_weight
 
        self.target_v_net.set_weights(target_net)
 
        return self.target_v_net    
    
        
    def memory_store(self,state, action, next_state, goal,proccessed_state,proccessed_n_state,G_reward,C_reward,traj_reward,d_reward):
    
        self.memory.append((state, action, next_state, goal,proccessed_state,proccessed_n_state,G_reward,C_reward,traj_reward,d_reward))
        return self.memory
     


        
    def put_net_together(self):
            st = self.backBone([self.state,self.goal])
            reward = self.reward_net(st)
            
            act_th , act_vel= self.Actor(st)
            
            model = Model(inputs=[self.state,self.goal],outputs=[act_th,act_vel,reward])
            return model 
            
        
    def train_Actor(self,state,goal,batch_size):
        
        with tf.GradientTape() as grad:
            st = self.backBone([state,goal])
            act = self.Actor(st)
            act=np.array(act).reshape(batch_size,2)
            
            act_th,act_vel,reward=self.model([state,goal])
            
            f_Q = tf.stop_gradient(self.q_net[0]([st,act]))
            sec_Q = tf.stop_gradient(self.q_net[1]([st,act]))
            
            Q =tf.minimum(sec_Q,f_Q)

            
            #print('Q === ',Q)

            action_loss= -tf.reduce_mean(Q - 0.1*tf.math.log(abs(act_th)+0.0001) - 0.1*tf.math.log(abs(act_vel)+0.0001))
            reward_loss = -tf.reduce_mean(0.0*reward + Q - 0.1*tf.math.log(abs(act_vel)+0.00001)- 0.1*tf.math.log(abs(act_th)+0.00001))         
        

        #print('[info] ############### ... training the main model ...##############')      
        model_gradients = grad.gradient([action_loss , reward_loss],self.model.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(model_gradients, self.model.trainable_weights))
        



def train(memory):
    if len(memory) >= 10:
        batch_size=10
    else:
        batch_size = len(memory)
    
    minibatch = random.sample(memory,batch_size)#??????
    minibatch = np.array(minibatch,dtype=object)
    st=[]
    act=[]
    n_st=[]
    goal = []
    pro_state= []
    pro_n_state = []
    G_reward = []
    C_reward = []
    traj_reward=[]
    d_reward  = []
    for i in range(len(minibatch)):
        st.append(minibatch[i][0])
        act.append(minibatch[i][1])
        n_st.append(minibatch[i][2])
        goal.append(minibatch[i][3])
        pro_state.append(minibatch[i][4][0])
        pro_n_state.append(minibatch[i][5][0])
        G_reward.append(minibatch[i][6])
        C_reward.append(minibatch[i][7])
        traj_reward.append(minibatch[i][8])
        d_reward.append(minibatch[i][9])

    st=np.array(st)
    n_st=np.array(n_st)
    act=np.array(act).reshape(batch_size,2)
    goal=np.array(goal)
    pro_state=np.array(pro_state)
    pro_n_state = np.array(pro_n_state)
    G_reward = np.array(G_reward)
    C_reward = np.array(C_reward)
    traj_reward = np.array(traj_reward)
    d_reward = np.array(d_reward)

    Lim=0.01
    sac.update_value_net(pro_state,act,batch_size)
    sac.update_D_Qn(pro_n_state,act,Lim,G_reward,C_reward,traj_reward,pro_state,d_reward)
    sac.train_Actor(st,goal,batch_size)
    sac.updateTargetModel()
    





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





# initializing the network_models
sac = SoftActorCritic(2,1024)




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
            Return = 0.0
            G_reward = 0.0
            C_reward = 0.0
            traj_reward = 0.0
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
                done_1 = contact_checker()
               
                current_pose = get_current_pose()
                relative_r_theta = np.array(relative_current_goal_pose(current_pose,np.array([x,y])))
                print('@@@@@@@@@@@@@@relative   ',relative_r_theta)
                #distance_saver.append(relative_r_theta[0]
                if done_1 == True:
                       # current_pose = get_current_pose()

                        state_2 = np.expand_dims(state,axis = 0)
                        goal_2 = np.expand_dims(relative_r_theta,axis=0)

                        output = sac.model([state_2,goal_2])
                        reward = output[2]
                        action = output[0:2]

                        proccessed_state = sac.backBone([state_2,goal_2])
                        print(f"Reward : {reward} &&&&& Episode : {episode}  &&&&& step : {t}")
                        Return = Return + reward

                        C_reward = -20.0

                        action= sac.sample_action(state_2,goal_2) # take the action and goes to the naxt state : observe the next state from the env
                        move.linear.x = action[0]
                        move.angular.z = action[1]
                        print(f'x ==> {action[0]} $ z ==> {action[1]}')
                        pub.publish(move)

                        current_pose = get_current_pose()
                        new_relative_r_theta = np.array(relative_current_goal_pose(current_pose,np.array([x,y])))

                        d_reward = dist_reward(200,new_relative_r_theta[0],relative_r_theta[0])

                        n_state=  get_image()  # it should be the actual state  ######### what the fuck     n_state = image_data
                        n_state_2 = np.expand_dims(n_state,axis = 0)
                        proccessed_n_state = sac.backBone([n_state_2,goal_2])
                        sac.memory_store(state,action,n_state,goal,proccessed_state,proccessed_n_state,G_reward,C_reward,traj_reward,d_reward) 

                        memory = sac.memory
                        train(memory)
                        print('episode has ended : We fucked an obstacle , Kir to hossein')
                       # print(f"Reward : {reward} & Goal : {goal}& Trajectory : {t}")
                       # print(f"total reward at episode {episode + 1} is {d_reward + C_reward + Return}")
                        break
                        

                elif  relative_r_theta[0] <= eps :


                        state_2 = np.expand_dims(state,axis = 0)
                        goal_2 = np.expand_dims(relative_r_theta,axis=0)

                        output = sac.model([state_2,goal_2])
                        reward = output[2]
                        action = output[0:2]

                        proccessed_state = sac.backBone([state_2,goal_2])
                        #print(f"Reward : {reward} & Goal : {goal}& Trajectory : {t}")
                        Return = Return + reward

                        G_reward = 100.0
                        
                        action= sac.sample_action(state_2,goal_2) # take the action and goes to the naxt state : observe the next state from the env
                        move.linear.x = action[0]
                        move.angular.z = action[1]
                        print(f'x ==> {action[0]} $ z ==> {action[1]}')
                        pub.publish(move)

                        d_reward = 0.0

                        n_state=  get_image()  # it should be the actual state  ######### what the fuck     n_state = image_data
                        n_state_2 = np.expand_dims(n_state,axis = 0)
                        proccessed_n_state = sac.backBone([n_state_2,goal_2])
                        sac.memory_store(state,action,n_state,goal,proccessed_state,proccessed_n_state,G_reward,C_reward,traj_reward,d_reward) 

                        memory = sac.memory
                        
                        train(memory)
                        print('episode has ended : We reached to our Goal , Kir to hossein')
                        
                        print(f"total reward at episode {episode + 1} is {G_reward + Return}")

                        break

                    

                else:


                        state_2 = np.expand_dims(state,axis = 0)
                        goal_2 = np.expand_dims(relative_r_theta,axis=0)

                        output = sac.model([state_2,goal_2])
                        reward = output[2]
                        action = output[0:2]

                        proccessed_state = sac.backBone([state_2,goal_2])
                       # print(f"Reward : {reward} & Goal : {goal}& Trajectory : {t}")
                        Return = Return + reward

                        action= sac.sample_action(state_2,goal_2) # take the action and goes to the naxt state : observe the next state from the env
                        move.linear.x = action[0]
                        move.angular.z = action[1]
                        print(f'x ==> {action[0]} $ z ==> {action[1]}')
                        pub.publish(move)

                        current_pose = get_current_pose()
                        new_relative_r_theta = np.array(relative_current_goal_pose(current_pose,np.array([x,y])))

                        d_reward = dist_reward(500,new_relative_r_theta[0],relative_r_theta[0])
                        #print('distance reward ======',dist_reward)
                        print(f"Reward : {reward} &&&&& Episode : {episode}  &&&&& step : {t}")

                        n_state=  get_image()  # it should be the actual state  ######### what the fuck     n_state = image_data
                        n_state_2 = np.expand_dims(n_state,axis = 0)
                        proccessed_n_state = sac.backBone([n_state_2,goal_2])
                        
                        if T < 100:
                            sac.memory_store(state,action,n_state,goal,proccessed_state,proccessed_n_state,G_reward,C_reward,traj_reward,d_reward) 
                        elif T == 100:
                             traj_reward = -10.0  # changed to 0 from -4 !!!!!!!!
                             sac.memory_store(state,action,n_state,goal,proccessed_state,proccessed_n_state,G_reward,C_reward,traj_reward,d_reward)

                        if len(sac.memory)>24:
                            memory = sac.memory
                            #print(memory)
                            train(memory)
                        
                                 
                                 
                        state=n_state

            print(f' #################[info] ... the Return of the trajectory {episode + 1} equals to {Return} #################')


    rospy.signal_shutdown(' :) Break :) ')


    
     

sac.model.save_weights('~/project/proj',overwrite=True)





#state_Dims
