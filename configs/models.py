import random

# Add reinforcement models


# Dummy class
class ReinforceAgent():
    def __init__(self, state_size, action_size):
        pass

    def trainModel(self):
        return

    def buildModel(self):
        return

    def get_action(self, action_space, state):
        #print("State: ", list(self.state_to_numeric(copy(state)).to_numpy().flatten()))
        action = random.choice(action_space)
        return action

    def appendMemory(self, former_state, new_state, action, reward, time_passed):
        return


rein_agent_1 = ReinforceAgent(70, 11)
rein_agent_2 = ReinforceAgent(140, 21)