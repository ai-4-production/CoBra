import os
import json
import numpy as np
import random
import time
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation

# Add reinforcement models


# Dummy class
class ReinforceAgent_1():
    def __init__(self, state_size, action_size):
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 40
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.batch_size = 8
        self.train_start = 8        
        self.memory = deque(maxlen=1000000)
        self.global_step = 0

        self.model = self.buildModel()
        self.target_model = self.buildModel()

        self.updateTargetModel()

        if self.load_model:
            self.model.set_weights(load_model(self.dirPath+str(self.load_episode)+".h5").get_weights())

            with open(self.dirPath+str(self.load_episode)+'.json') as outfile:
                param = json.load(outfile)
                self.epsilon = param.get('epsilon')

    
    def buildModel(self):
        model = Sequential()
        dropout = 0.1
        model.add(Dense(32, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()
        return model

    def trainModel(self, target):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.int)
        Y_batch = np.empty((0, self.action_size), dtype=np.int)
        for i in range(self.batch_size):
            states = np.asarray(mini_batch[i][0])
            next_states = np.asarray(mini_batch[i][3])
            actions = np.asarray(mini_batch[i][1])
            rewards = np.asarray(mini_batch[i][2])
            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value
            next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))
                self.updateTargetModel()

            #missing update target model
            #save model option

            next_q_value = self.getQvalue(rewards, next_target)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()
            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)

    def getQvalue(self, reward, next_target):
        return reward + self.discount_factor * np.amax(next_target)

    def get_action(self, state):
        self.global_step = self.global_step + 1
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        epsilon_random = np.random.rand()
        
        if epsilon_random <= self.epsilon:
            Smart_action = False
            action = random.randint(0, self.action_size)
            return action, Smart_action
            # return random.randrange(self.action_size)
        else:
            Smart_action = True
            state = np.array(state)
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            return q_value, Smart_action

    def appendMemory(self, smart_agent, former_state, new_state, action, reward, time_passed):
        #smart_agent, former_state=old_state_flat, new_state=new_state_flat, action=action, reward=reward, time_passed=time_passed
        self.memory.append((former_state, action, reward, new_state))
        #if len(smart_agent.memory) >= smart_agent.train_start:
        if (smart_agent.global_step % smart_agent.target_update) == 0:
            smart_agent.updateTargetModel()
        if len(smart_agent.memory) >= smart_agent.batch_size:
            smart_agent.trainModel(True)



    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())


rein_agent_1 = ReinforceAgent(11, 3)  #current one
