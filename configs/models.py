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
class ReinforceAgent():
    def __init__(self, state_size, action_size):
        self.load_model = False
        self.load_episode = 0
        self.state_size = state_size
        self.action_size = action_size
        self.episode_step = 6000
        self.target_update = 2000
        self.discount_factor = 0.99
        self.learning_rate = 0.01
        self.epsilon = 1.0
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.05
        self.batch_size = 64
        self.train_start = 64
        self.memory = deque(maxlen=1000000)

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
        dropout = 0.2
        model.add(Dense(64, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(lr=self.learning_rate, rho=0.9, epsilon=1e-06))
        model.summary()
        return model
        return

    def trainModel(self, target=False):
        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=np.int)
        Y_batch = np.empty((0, self.action_size), dtype=np.int)
        for i in range(self.batch_size):
            states = np.asarray(mini_batch[i][0])
            next_states = np.asarray(mini_batch[i][1])
            actions = np.asarray(mini_batch[i][2])
            rewards = np.asarray(mini_batch[i][3])

            q_value = self.model.predict(states.reshape(1, len(states)))
            self.q_value = q_value

            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)))

            else:
                next_target = self.model.predict(next_states.reshape(1, len(next_states)))

            #missing update target model
            #save model option

            next_q_value = self.getQvalue(rewards, next_target)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()

            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)


        


    def getQvalue(self, reward, next_target):
        if done:
            return reward
        else:
            return reward + self.discount_factor * np.amax(next_target)

    def get_action(self, action_space, state):
        #print("State: ", list(self.state_to_numeric(copy(state)).to_numpy().flatten()))
        action = random.choice(action_space)
        print("RL")

        if np.random.rand() <= self.epsilon:
            self.q_value = np.zeros(self.action_size)
            # return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state.reshape(1, len(state)))
            self.q_value = q_value
            # return np.argmax(q_value[0])
        return action    

    def appendMemory(self, former_state, new_state, action, reward, time_passed):
        self.memory.append((former_state, action, reward, new_state))
         if len(agent.memory) >= agent.train_start:
            if agent.global_step <= agent.target_update:
                agent.trainModel()
            else:
                agent.trainModel(True)



    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())


rein_agent_1 = ReinforceAgent(70, 11)
rein_agent_1_1 = ReinforceAgent(140, 21)
rein_agent_1_2 = ReinforceAgent(140, 21)