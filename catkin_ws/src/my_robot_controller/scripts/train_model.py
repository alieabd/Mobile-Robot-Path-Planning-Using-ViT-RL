#! /usr/bin/python3
import numpy as np


import rospy
from std_srvs.srv import Empty


from tensorflow.keras.layers import Dropout,Dense,Flatten,Input,Activation,concatenate
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from collections import deque


goal = np.load("dummy_goal.npy")




M=70
T=60
#sig=0.5
batch_size=10
gamma=0.01
alpha=0.3
#a_1 = 0.5
#a_2 = 0.5



input_shape = (160, 120, 3)
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 50
image_size = 256  # We'll resize input images to this size
patch_size = 32  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
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
    


def Double_Q_network(s,a,Neurons):
    state= Input(shape=(s,))
    action=Input(shape=(a,))
    
    x_1 = Dense(Neurons, activation="relu")(state)
    x_1 = Dense(40, activation="relu")(x_1)
   
    x_2 = Dense(Neurons, activation="relu")(action)
    x_2 = Dense(40, activation="relu")(x_2)
    
    x=concatenate([x_1,x_2],axis=-1)
    
    theta_Q=Dense(20, activation="relu")(x)
    theta_Q=Dense(1, activation="relu")(theta_Q)
    
    theta_net =Activation("linear", name="theta_output")(theta_Q)
    
    velocity_Q=Dense(20, activation="relu")(x)
    velocity_Q=Dense(1, activation="relu")(velocity_Q)
    
    velocity_net =Activation("linear", name="vel_output")(velocity_Q)
    
    losses={'theta_output':'mse','vel_output':'mse'}
    loss_weight = {"theta_output": 1.0, "vel_output": 1.0}
    
    Q_1 = Model(inputs = [state,action],outputs = [theta_net,velocity_net],name='first_Q')
    
    Q_1.compile(optimizer=Adam(learning_rate = alpha), loss=losses,loss_weights = loss_weight)# a dict must be passed to loss
    
    y_1 = Dense(Neurons, activation="relu")(state)
    y_1 = Dense(40, activation="relu")(y_1)
   
    y_2 = Dense(Neurons, activation="relu")(action)
    y_2 = Dense(40, activation="relu")(y_2)

    y=concatenate([y_1,y_2],axis=-1)
    
    theta_Q_2=Dense(20, activation="relu")(y)
    theta_Q_2=Dense(1, activation="relu")(theta_Q_2)
    
    theta_net_2 =Activation("linear", name="theta2_output")(theta_Q_2)
    
    velocity_Q_2=Dense(20, activation="relu")(y)
    velocity_Q_2=Dense(1, activation="relu")(velocity_Q_2)
    
    velocity_net_2 =Activation("linear", name="vel2_output")(velocity_Q_2)
    
    losses_2={'theta2_output':'mse','vel2_output':'mse'}
    loss_weight_2 = {"theta2_output": 1.0, "vel2_output": 1.0}
    
    Q_2 = Model(inputs = [state,action],outputs = [theta_net_2,velocity_net_2],name='second_Q')
    
    Q_2.compile(optimizer=Adam(learning_rate = alpha), loss=losses_2,loss_weights = loss_weight_2)
    
    return Q_1,Q_2




def feature_ext():
    image_inputs = Input(shape=input_shape) #shape=(None, 32, 32, 3)
    goal_inputs = Input(shape=(2,))
    # Augment data.
    augmented = data_augmentation(image_inputs) #shape=(None, 72, 72, 3)
    # Create patches.
    patches = Patches(patch_size)(augmented) #shape=(None, None, 108)
    
    #Add Goal to Patches
    Goal = mlp(goal_inputs,[32,512,3072],dropout_rate=0.05) #shape=(1, 192)
    Goal = tf.reshape(Goal,[1,1,3072])

    tf.keras.layers.Concatenate(axis=1)([Goal,patches])    
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches) #shape=(None, 144, 64)
    
    
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
    angular_vel = layers.Dense(1,)(linear_vel)                        
    angular_net =Activation("linear", name="angular_output")(angular_vel)
    
    model = Model(inputs = features , outputs = [linear_vel,angular_vel])
    
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
    angular_vel = Dense(1,)(linear_vel)                        
    angular_net =Activation("linear", name="angular_output")(angular_vel)
    
    ConCat = concatenate([linear_net , angular_net] , axis = -1)
    
    x=Dense(32,activation="relu")(ConCat)
    x=Dense(16,activation="relu")(x)
    x=Dense(1)(x)
    
    out=Activation("linear",name="rew_out")(x)
    
    
    reward_net = Model(inputs=features,outputs=out,name="TPr_net")
    
    return reward_net




class SoftActorCritic:

    def __init__(self, action_dim,state_dim, epoch_step=1, learning_rate=0.0003,alpha=0.2, gamma=0.99):
        self.state = Input(shape = (160,120,3)) 
        self.goal =  Input(shape = (2,))
        self.Actor = Actor(state_dim)
        self.q_net = Double_Q_network(state_dim,action_dim,30)  
        self.v_net = value_net(state_dim,10)
        self.target_v_net = target_value_net(state_dim,10)
        self.backBone = feature_ext()
        self.reward_net = trainable_reward_func(state_dim) 
        self.log_dir = '~/project/proj/'
        
        self.backBone.load_weights(self.log_dir + 'Back_bone.h5')
        self.Actor.load_weights(self.log_dir + 'actor.h5')
        self.reward_net.load_weights(self.log_dir + 'reward.h5') 
        self.model  = self.put_net_together()
        
        self.memory = memory=deque([], maxlen=2500)
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

    def update_D_Qn(self,n_state,action):
        n_st_value = self.target_v_net(n_state)
        target_Q = self.reward_net(n_state) + n_st_value
        
        print('############[info] ... training Q (critic) networks ... ###########################3')
        self.q_net[0].fit(x=[n_state,action], y={'theta_output':target_Q,'vel_output':target_Q}, epochs = self.epoch,verbose=0)
        self.q_net[1].fit(x=[n_state,action], y={'theta2_output':target_Q,'vel2_output':target_Q}, epochs = self.epoch,verbose=0)
        
    def update_value_net(self,st,act):
            first_Q_Double=tf.stop_gradient(self.q_net[0].predict([st,act]))
            second_Q_Double=tf.stop_gradient(self.q_net[1].predict([st,act]))

            Q_th_f=first_Q_Double[0][:]
            Q_vel_f=first_Q_Double[1][:]


            Q_th_sec=second_Q_Double[0][:]
            Q_vel_sec=second_Q_Double[1][:]


            Q_th=tf.minimum(Q_th_f,Q_th_sec)
            Q_vel=tf.minimum(Q_vel_f,Q_th_sec)
            
            act_th = act[0:batch_size,0].reshape(batch_size,1)
            act_vel = act[0:batch_size,1].reshape(batch_size,1)
            
            act_th = abs(act_th) + 0.0001
            act_vel = abs(act_vel) + 0.0001
            
            bel_back_th = Q_th - np.log10(act_th)
            bel_back_vel = Q_vel - np.log10(act_vel)
            
            res = tf.add(Q_th,Q_vel)
            bel_back = tf.multiply(tf.constant([0.5]) , res)
            print('[info] ################### ... training value network .... ###################')
            self.v_net.fit(st, bel_back,epochs=1, verbose=0)
           

    def updateTargetModel(self,tau=0.2):
        weights= self.v_net.get_weights()
        target_net = self.target_v_net.get_weights()
        
        for i in range(len(weights)):

            new_weight = weights[i]+ (1-tau)*target_net[i]
            target_net[i] = new_weight
 
        self.target_v_net.set_weights(target_net)
 
        return self.target_v_net    
    
        
    def memory_store(self,state, action, next_state, goal,proccessed_state):
    
        self.memory.append((state, action, next_state, goal,proccessed_state))
        return self.memory
     


        
    def put_net_together(self):
            st = self.backBone([self.state,self.goal])
            reward = self.reward_net(st)
            
            act_th , act_vel= self.Actor(st)
            
            model = Model(inputs=[self.state,self.goal],outputs=[act_th,act_vel,reward])
            return model 
            
        
    def train_Actor(self,state,goal):
        
        with tf.GradientTape() as grad:
            st = self.backBone([state,goal])
            act = self.Actor(st)
            act=np.array(act).reshape(10,2)
            
            act_th,act_vel,reward=self.model([state,goal])
            
            
            
            
            first_Q_Double=tf.stop_gradient(self.q_net[0]([st,act]))
            second_Q_Double=tf.stop_gradient(self.q_net[1]([st,act]))
            
            Q_th_f=first_Q_Double[0][:]
            Q_vel_f=first_Q_Double[1][:]


            Q_th_sec=second_Q_Double[0][:]
            Q_vel_sec=second_Q_Double[1][:]


            Q_th=tf.minimum(Q_th_f,Q_th_sec)
            Q_vel=tf.minimum(Q_vel_f,Q_th_sec)
            

            action_loss= -tf.reduce_mean(tf.multiply(tf.constant([0.5]),Q_th - tf.math.log(act_th+0.0001) + Q_vel - tf.math.log(act_vel+0.0001)))
            reward_loss = -tf.reduce_mean(tf.multiply(tf.constant([0.5]), reward + Q_th + Q_vel - tf.math.log(act_vel+0.0001)- tf.math.log(act_th+0.0001)))            
        

        print('[info] ############### ... training the main model ...##############')      
        model_gradients = grad.gradient([action_loss , reward_loss],self.model.trainable_weights)
        self.actor_optimizer.apply_gradients(zip(model_gradients, self.model.trainable_variables))
      #  print(model_gradients)
      
     #   return model
        



def train(memory , batch_size=10):
    
    minibatch = random.sample(memory,batch_size)
    minibatch = np.array(minibatch,dtype=object)
    st=[]
    act=[]
    n_st=[]
    goal = []
    pro_state= []
    for i in range(len(minibatch)):
        st.append(minibatch[i][0])
        act.append(minibatch[i][1])
        n_st.append(minibatch[i][2])
        goal.append(minibatch[i][3])
        pro_state.append(minibatch[i][4][0])
        
    st=np.array(st)
    n_st=np.array(n_st)
    act=np.array(act).reshape(batch_size,2)
    goal=np.array(goal)
    pro_state=np.array(pro_state)

    sac.update_value_net(pro_state,act)
    sac.update_D_Qn(pro_state,act)
    sac.train_Actor(st,goal)
    sac.updateTargetModel()
    
    
    



# initializing the network_models
sac = SoftActorCritic(2,1024)



for episode in range(M):
        Return = 0

        rospy.wait_for_service('/gazebo/reset_world')
        reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
        reset_world()

        state = image_data
        # Here you should define youre environment
        # Defining the initial state 
        # including reset function
        
        
        for t in range(T):
            
            state= sac.sample_state() # sample state take the states from actual scene : this part must be comented
            goal = sac.random_goal()  # fake goal we need a real one
            
            state_2 = np.expand_dims(state,axis = 0)
            goal_2 = np.expand_dims(goal,axis=0)
            
            proccessed_state = sac.backBone([state_2,goal_2])
            reward = sac.reward_net(proccessed_state)
            
            Return = Return + reward
            
            action= sac.sample_action(state_2,goal_2) # take the action and goes to the naxt state : observe the next state from the env
            n_state= sac.sample_state()  # it should be the actual state
            sac.memory_store(state,action,n_state,goal,proccessed_state) 
            

            
            if len(sac.memory)>batch_size:
                memory = sac.memory
                train(memory)
            state=n_state
    
        print(f' #################[info] ... the Return of the trajectory {episode + 1} equals to {Return} #################')














    


rospy.spin()

