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
from keras.models import Sequential, load_model
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Activation
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
        self.target_update = 5
        self.discount_factor = 0.98 #previous: 0.999
        self.learning_rate = 0.0005 #previous: 0.005
        self.epsilon = 1.0
        self.epsilon_decay = 0.997 #previous: 0.999
        self.epsilon_min = 0.01
        self.batch_size = 128
        self.memory = deque(maxlen=1000000)
        self.global_step = 0
        self.global_step_1 = 0
        self.train_step = 0
        self.train_step_1 = 0
        self.save_rate = 50
        self.rewards = []
        self.rewards_average_old = -1000
        t = time.localtime()
        self.timestamp = time.strftime('%Y-%m-%d_%H-%M', t)

        self.model = self.buildModel()
        self.target_model = self.buildModel()
        self.path = pathlib.Path(__file__).parent.resolve()

        if self.operational_mode:
            try:
                self.model = load_model("models_saved/best_models/" + str(self.identifier))
            except:
                self.model = self.buildModel()
                self.target_model = self.buildModel()    
        else:
            self.model = self.buildModel()
            self.target_model = self.buildModel()

    def buildModel(self):
        model = Sequential()
        dropout = 0.01
        model.add(Dense(self.hidden_layer_size_1, input_shape=(self.state_size,), activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='lecun_uniform'))
        model.add(Dropout(dropout))
        model.add(Dense(self.action_size, kernel_initializer='lecun_uniform'))
        model.add(Activation('linear'))
        model.compile(loss='mse', optimizer=RMSprop(learning_rate=self.learning_rate, rho=0.9, epsilon=1e-06))
        # model.summary()
        return model

    def trainModel(self, target):
        now_0 = time.time()

        # with open('../models_saved/memories/memory_' + str(self.identifier) + '.txt', 'r', newline='', encoding='utf-8') as f:
        #     reader = csv.reader(f)
        #     memory = list(reader)
        # data = np.loadtxt('../models_saved/memories/memory_' + str(self.identifier) + '.txt', delimiter=',')
        # mini_batch = random.sample(memory, self.batch_size)
        # mini_batch = random.sample(data, self.batch_size)

        mini_batch = random.sample(self.memory, self.batch_size)
        X_batch = np.empty((0, self.state_size), dtype=int)
        Y_batch = np.empty((0, self.action_size), dtype=int)
        for i in range(self.batch_size):
            states = np.asarray(mini_batch[i][0])
            next_states = np.asarray(mini_batch[i][3])
            actions = np.asarray(mini_batch[i][1])
            rewards = np.asarray(mini_batch[i][2])
            q_value = self.model.predict(states.reshape(1, len(states)), verbose = 0)
            self.q_value = q_value
            next_target = self.model.predict(next_states.reshape(1, len(next_states)), verbose = 0)
            if target:
                next_target = self.target_model.predict(next_states.reshape(1, len(next_states)), verbose = 0)
            # K.clear_session()
            next_q_value = self.getQvalue(rewards, next_target)

            X_batch = np.append(X_batch, np.array([states.copy()]), axis=0)
            Y_sample = q_value.copy()
            Y_sample[0][actions] = next_q_value
            Y_batch = np.append(Y_batch, np.array([Y_sample[0]]), axis=0)
            tf.keras.backend.clear_session()
        self.model.fit(X_batch, Y_batch, batch_size=self.batch_size, epochs=1, verbose=0)
        
        time_tracker.time_train_calc = time.time() - now_0
        gc.collect()
        # K.clear_session()

    def getQvalue(self, reward, next_target):
        return reward + self.discount_factor * np.amax(next_target)

    def get_dispatch_rule(self, state):
        self.global_step = self.global_step + 1
        if not self.operational_mode:
            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay
                
            epsilon_random = np.random.rand()
            if epsilon_random <= self.epsilon:
                action = random.randint(0, self.action_size - 1)
                return action
                # return random.randrange(self.action_size)
            else:
                state = np.array(state)
                q_value = self.model.predict(state.reshape(1, len(state)), verbose = 0)
                self.q_value = q_value
                action = np.argmax(self.q_value[0])
                return action

        elif self.operational_mode:
            state = np.array(state)
            q_value = self.model.predict(state.reshape(1, len(state)), verbose = 0)
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
                    self.model.save("../models_saved/all_models/" + str(self.identifier) + "_" + self.timestamp + "_" + str(self.train_step))
                    print("model_saved")
                    # self.model.save("../models_saved/all_models/" + self.timestamp + "_" + 'cell.id-' + str(cell_id) +  '_' + str(self) + "_" + str(self.action_size) + "_" + str(self.state_size) + "_" + str(self.hidden_layer_size_1) + "_" + str(self.batch_size) + "_" + str(self.train_step))

    def updateTargetModel(self):
        self.target_model.set_weights(self.model.get_weights())

# rein_agent_dispatch_0 = ReinforceAgent(57, 3) #current one
# rein_agent_dispatch_1 = ReinforceAgent(48, 3) #current one

# scenario within paper
# rein_agent_dispatch_scenario_paper_d_1 = ReinforceAgent(88, 5, False)
# rein_agent_dispatch_scenario_paper_d_1_1 = ReinforceAgent(52, 5, False)
# rein_agent_dispatch_scenario_paper_d_1_2 = ReinforceAgent(84, 5, False)
# rein_agent_dispatch_scenario_paper_m_1_1_1 = ReinforceAgent(80, 5, False)
# rein_agent_dispatch_scenario_paper_m_1_1_2 = ReinforceAgent(60, 5, False)

# rein_agent_dispatch_4 = ReinforceAgent(128, 5, False) #current one
# rein_agent_dispatch_4 = ReinforceAgent(81, 3, True, 6400) #current one

#operations of scenario_1
# # rein_agent_dispatch_scenario_paper_d_1 = ReinforceAgent(88, 5, True, 1)
# rein_agent_dispatch_scenario_paper_d_1_1 = ReinforceAgent(84, 5, True, 1)
# # #rein_agent_dispatch_scenario_paper_d_1_2 = ReinforceAgent(52, 5, True, 1)
# rein_agent_dispatch_scenario_paper_m_1_1_1 = ReinforceAgent(80, 5, True, 1)
# rein_agent_dispatch_scenario_paper_m_1_1_2 = ReinforceAgent(60, 5, True, 1)

#scenario_FAZI
# rein_agent_dispatch_scenario_paper_d_1 = ReinforceAgent(124, 5, True, 1)
# # rein_agent_dispatch_scenario_paper_d_1_1 = ReinforceAgent(52, 5, False)
# # rein_agent_dispatch_scenario_paper_d_1_2 = ReinforceAgent(84, 5, False)
# # rein_agent_dispatch_scenario_paper_m_1_1_1 = ReinforceAgent(80, 5, False)
# # rein_agent_dispatch_scenario_paper_m_1_1_2 = ReinforceAgent(68, 5, False)
