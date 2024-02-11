import gym
import tinycarlo

env = gym.make("tinycarlo-v1", config="./example/segmented_map/")

observation = env.reset()
while True:
    env.render()
    action = env.action_space.sample()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
        break
env.close()