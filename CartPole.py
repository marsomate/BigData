
# Deep Q-learning CartPole example
# based on https://github.com/gsurma/cartpole.git & https://gym.openai.com/evaluations/eval_OeUSZwUcR2qSAqMmOE1UIw/

import gym
import random
import numpy as np
from os import path
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

GAMMA = 0.99
LEARNING_RATE = 0.001

MEMORY_SIZE = 1000
BATCH_SIZE = 64

EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.1
EXPLORATION_DECAY = 0.99

env = gym.make("CartPole-v0")
observation_space_size = env.observation_space.shape[0]
action_space_size = env.action_space.n

model = Sequential()
model.add(Dense(16, input_shape=(observation_space_size, ), activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(action_space_size, activation='linear'))
model.summary()

#if path.exists("cartpole_weights.index"):
#    print("Loading weights from cartpole_weights...")
#    model.load_weights("cartpole_weights")

model.compile(loss="mse", optimizer=Adam(lr=LEARNING_RATE))

exploration_rate = EXPLORATION_MAX
memory = deque(maxlen=MEMORY_SIZE)
run = 0

while True:
    run += 1
    observation = env.reset()
    step = 0
    while True:
        step += 1
        # env.render()

        if np.random.rand() < exploration_rate:
            action = random.randrange(action_space_size)
        else:
            state = np.reshape(observation, [1, observation_space_size])
            q_values = model.predict(state)
            action = np.argmax(q_values[0])

        observation_next, reward, done, info = env.step(action)
        reward = reward if not done else -200
        memory.append((observation, action, reward, observation_next, done))
        observation = observation_next

        if done:
            print("Run: " + str(run) + ", exploration: " + str(exploration_rate) + ", score: " + str(step))
            
            if len(memory) >= BATCH_SIZE:
                state_batch, qvalue_batch = [], []
                batch = random.sample(memory, BATCH_SIZE)
                for observation, action, reward, observation_next, done in batch:
                    q_update = reward
                    if not done:
                        state_next = np.reshape(observation_next, [1, observation_space_size])
                        q_predicted = np.amax(model.predict(state_next)[0])
                        q_update = reward + (GAMMA * q_predicted)
                    state = np.reshape(observation, [1, observation_space_size])
                    q_values = model.predict(state)
                    q_values[0][action] = q_update
                    state_batch.append(state[0])
                    qvalue_batch.append(q_values[0])
                model.fit(np.array(state_batch), np.array(qvalue_batch), batch_size=len(state_batch), epochs=1, verbose=0)
                #model.save_weights("cartpole_weights")
        
                exploration_rate *= EXPLORATION_DECAY
                exploration_rate = max(EXPLORATION_MIN, exploration_rate)

            break
