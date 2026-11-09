import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make("CartPole-v1", render_mode = "rgb_array")

initial_state, info = env.reset(seed = 42)

def render():
    state_image = env.render()
    plt.imshow(state_image)
    plt.show()

terminated = False

while not terminated:
    action = 1
    state, reward, terminated, truncated, info = env.step(action)
    render()

print("State:", state)
print("Info:", info)
print("Terminated:", terminated)
