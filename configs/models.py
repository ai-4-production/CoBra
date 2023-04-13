import os
import stat
import gc
import json
import numpy as np
import random
import time
import pathlib
import csv
from utils import time_tracker
import sys
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
import tensorflow as tf
from keras.models import Sequential, load_model, Model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation, Lambda, Input
os.environ['keras_backend'] = 'tensorflow'
from keras.backend import backend as K
from keras.callbacks import History 
history = History()

# Add reinforcement models
#save model option

class ReinforceAgent():
    def __init__(self, state_size, action_size, operational_mode = False, identifier = None):
        self.identifier = identifier
        self.operational_mode = operational_mode
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.hidden_layer_size_1 = 128
        self.target_update = 3
        self.discount_factor = 0.98 #previous: 0.999
        self.learning_rate = 0.005 #previous: 0.999
        self.epsilon_min = 0.01
        self.epsilon = 1
        self.epsilon_decay = 0.997
        self.batch_size = 128
        self.memory = deque(maxlen=1000000)
        self.global_step = 0
        self.global_step_1 = 0
        self.train_step = 0
        self.train_step_1 = 0
        self.save_rate = 100
        self.rewards = []
        self.rewards_average_old = -1000
        t = time.localtime()
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M', t)

        self.dqn = False
        self.transfer_weights = False
        self.double_DQN = True
        self.dueling_DQN = False

        self.model = self.buildModel(self.dqn)
        self.target_model = self.buildModel(self.dqn)
        self.path = pathlib.Path(__file__).parent.resolve()

        if self.operational_mode:
            try:
                self.model = load_model("../models_saved/best_models/" + str(self.identifier))
                print("Neural network model found for agent with ID: ", self.identifier)
                time.sleep(1)
            except:
                self.model = self.buildModel(self.dqn)
                self.target_model = self.buildModel(self.dqn)
                print("No network found for agent with ID: ", self.identifier) 
                print("Operational mode is switched to training mode") 
                self.operational_mode = False  
        else:
            self.model = self.buildModel(self.dqn)
            self.target_model = self.buildModel(self.dqn)

    def buildModel(self, dqn = False):
        # define a plain DQN neural network structure as a feed forward neural network
        model = Sequential()
        if self.dqn == True:
            print("Compiled DQN model")
            dropout = 0.01

            # add input, output and hidden layers
            model.add(Dense(self.hidden_layer_size_1, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
            model.add(Dense(128, activation='relu', kernel_initializer='lecun_uniform'))
            model.add(Dropout(dropout))
            model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
            model.add(Activation('linear'))
            model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06))
            
        # define a dueling DQN neural network structure with an advantage and critic network
        if self.dueling_DQN == True:
            if self.double_DQN == True:
                print("Compiled dueling double DQN model")
            else: 
                print("Compiled dueling DQN model")
            input_states = Input(shape=(self.state_size,))

            # Common hidden layers
            x = Dense(self.hidden_layer_size_1, activation='relu', kernel_initializer='lecun_uniform')(input_states)
            x = Dense(128, activation='relu', kernel_initializer='lecun_uniform')(x)
            x = Dropout(0.01)(x)

            # Value stream
            value = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)
            value = Dense(1, kernel_initializer='lecun_uniform')(value)

            # Advantage stream
            advantage = Dense(64, activation='relu', kernel_initializer='lecun_uniform')(x)
            advantage = Dense(self.action_size, kernel_initializer='lecun_uniform')(advantage)
            advantage = Lambda(lambda a: a - tf.reduce_mean(a, axis=-1, keepdims=True))(advantage)

            # Combine value and advantage streams
            q_values = Lambda(lambda v_a: v_a[0] + v_a[1])([value, advantage])

            model = Model(inputs=input_states, outputs=q_values)
            model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06))
        
        # transfer of neural network weights
        if self.transfer_weights and self.state_size == 150:
            self.epsilon = 0.5
            transferred_layers = 0
            similar_model = load_model("../models_saved/best_models/" + str(self.identifier))
            print("Transferred network parameters")
            for i, layer in enumerate(model.layers[1:-2]):  # Excluding the input and last layer
                if isinstance(layer, Dense):
                    layer.set_weights(similar_model.layers[i].get_weights())
                    transferred_layers += 1
                    if transferred_layers >= 2:
                        break
    
        return model    
            
    def trainModel(self, target):
        now_0 = time.time()  # Record start time

        # with open('../models_saved/memories/memory_' + str(self.identifier) + '.txt', 'r', newline='', encoding='utf-8') as f:
        #     reader = csv.reader(f)
        #     memory = list(reader)
        # data = np.loadtxt('../models_saved/memories/memory_' + str(self.identifier) + '.txt', delimiter=',')
        # mini_batch = random.sample(memory, self.batch_size)
        # mini_batch = random.sample(data, self.batch_size)

         # Sample a mini-batch of experiences from the replay memory
        mini_batch = random.sample(self.memory, self.batch_size)

        # Initialize input (states) and output (target Q-values) batches
        X_batch = np.empty((0, self.state_size), dtype=int)
        Y_batch = np.empty((0, self.action_size), dtype=int)

        # Iterate through each experience in the mini-batch
        for i in range(self.batch_size):
            states = np.asarray(mini_batch[i][0])
            next_states = np.asarray(mini_batch[i][3])
            actions = np.asarray(mini_batch[i][1])
            rewards = np.asarray(mini_batch[i][2])

            # Get the current Q-values for the given state
            q_value = self.model.predict(states.reshape(1, len(states)), verbose=0)

            # Get the target Q-values for the next state
            next_target = self.target_model.predict(next_states.reshape(1, len(next_states)), verbose=0)

            # Get the best action according to the online model
            best_online_action = np.argmax(self.model.predict(next_states.reshape(1, len(next_states)), verbose=0))

            # Calculate the target Q-value for the selected action (plain or double DQN)
            next_q_value = self.getQvalue(rewards, next_target, best_online_action)

            # Add the current state to the input batch
            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)

            # Add the updated Q-value to the output batch
            Y_sample = q_value.copy()
            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

            # Clear the TensorFlow session to avoid memory leaks
            tf.keras.backend.clear_session()

        # Train the model using the input and output batches
        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)

        # Calculate and store the training time
        time_tracker.time_train_calc = time.time() - now_0

        # Clear memory
        gc.collect()
        # K.clear_session()

    def getQvalue(self, reward, next_target, best_online_action):
        if self.dqn == True:
            # Calculate the target Q-value using standard Q-learning
            return reward + self.discount_factor * np.amax(next_target)
        if self.double_DQN == True:
            # Calculate the target Q-value using Double Q-learning
            return reward + self.discount_factor * next_target[0][best_online_action]

    def get_dispatch_rule(self, state):
        self.global_step = self.global_step + 1

        if not self.operational_mode:
            # Decay the epsilon value if it's greater than the minimum value
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay

            # Determine whether to take a random action based on the epsilon value
            epsilon_random = np.random.rand()
            if epsilon_random <= self.epsilon:
                # Take a random action
                action = random.randint(0, self.action_size - 1)
                return action
            else:
                # Take the best action according to the current Q-values
                state = np.array(state)
                q_value = self.model.predict(state.reshape(1, len(state)), verbose=0)
                self.q_value = q_value
                action = np.argmax(self.q_value[0])
                return action
            
        elif self.operational_mode:
            # In operational mode, always take the best action according to the current Q-values
            state = np.array(state)
            q_value = self.model.predict(state.reshape(1, len(state)), verbose=0)
            self.q_value = q_value
            action = np.argmax(self.q_value[0])
            return action

    def appendMemory(self, smart_agent, cell_id, former_state, new_state, action, reward):
        self.memory.append((former_state, action, reward, new_state))
        smart_agent.rewards.append(reward)
        # with open('../models_saved/memories/memory_' + str(self.identifier) + '.txt', 'a+', newline='', encoding='utf-8') as f:
        #     writer = csv.writer(f)
        #     writer.writerow((former_state, action, reward, new_state))

        if not self.operational_mode:
            if (smart_agent.global_step % smart_agent.target_update) == 0:
                smart_agent.updateTargetModel()
            if len(smart_agent.memory) >= self.batch_size:
                smart_agent.trainModel(True)
                self.train_step += 1
                smart_agent.train_step_1 += 1
                if self.train_step % self.save_rate == 0:
                    print(np.average(smart_agent.rewards))
                    # if self.rewards_average_old <= np.average(smart_agent.rewards):
                    #     os.chdir("..")
                    #     os.chmod(str(os.path.abspath(os.curdir)) + "/models_saved/best_models/" + str(self.identifier), stat.S_IWUSR | stat.S_IRUSR)
                    #     os.remove(str(os.path.abspath(os.curdir)) + "/models_saved/best_models/" + str(self.identifier))
                    #     print(self.path)
                    #     self.model.save("../models_saved/best_models/" + str(self.identifier))
                    #     self.rewards_average_old = np.average(self.rewards)
                    #     self.rewards = []
                    #     print("Best_model in cell ", cell_id, " updated")
                    if self.dueling_DQN:
                        self.model.save("../models_saved/all_models/dueling_" + str(self.identifier) + "_" + self.timestamp + "_" + str(self.train_step))
                    else:
                        self.model.save("../models_saved/all_models/" + str(self.identifier) + "_" + self.timestamp + "_" + str(self.train_step))
                    print("model_saved")
                    # self.model.save("../models_saved/all_models/" + self.timestamp + "_" + 'cell.id-' + str(cell_id) +  '_' + str(self) + "_" + str(self.action_size) + "_" + str(self.state_size) + "_" + str(self.hidden_layer_size_1) + "_" + str(self.batch_size) + "_" + str(self.train_step))

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())