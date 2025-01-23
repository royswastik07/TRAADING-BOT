import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers, models

class ReplayBuffer:
    """
    Replay buffer for storing experiences.
    """
    def __init__(self, max_size=100000):
        self.buffer = []
        self.max_size = max_size

    def add(self, experience):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

def build_q_network(state_shape, action_space):
    """
    Build a Q-network model.
    """
    model = models.Sequential()
    model.add(layers.Input(shape=state_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

class DQNAgent:
    """
    Deep Q-Network agent.
    """
    def __init__(self, state_shape, action_space, gamma=0.99, epsilon=1.0, 
                 epsilon_decay=0.995, epsilon_min=0.01, batch_size=32):
        self.state_shape = state_shape
        self.action_space = action_space
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.q_network = build_q_network(state_shape, action_space)
        self.target_network = build_q_network(state_shape, action_space)
        self.update_target_network()
        self.replay_buffer = ReplayBuffer()

    def update_target_network(self):
        self.target_network.set_weights(self.q_network.get_weights())

    def choose_action(self, state):
        state = np.reshape(state, (1, *self.state_shape))
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_space)
        q_values = self.q_network.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = np.array(states).reshape((self.batch_size, *self.state_shape))
        next_states = np.array(next_states).reshape((self.batch_size, *self.state_shape))
        target_q_values = self.q_network.predict(states, verbose=0)
        next_q_values = self.target_network.predict(next_states, verbose=0)
        for i in range(self.batch_size):
            if dones[i]:
                target_q_values[i][actions[i]] = rewards[i]
            else:
                target_q_values[i][actions[i]] = rewards[i] + self.gamma * np.max(next_q_values[i])
        self.q_network.fit(states, target_q_values, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
