import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

env = gym.make("ALE/Breakout-v5")

epochs = 0

frames = []  # for animation
done = False

env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
observation, info = env.reset()

while not done:
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Put each rendered frame into dict for animation
    frames.append(
        {
            "frame": env.render(),
            "state": observation,
            "action": action,
            "reward": reward,
        }
    )

    epochs += 1
    if epochs == 1000:
        break

from PIL import Image

def create_gif(frames: list[dict], filename, fps=100):
    """
    Creates a GIF animation from the image frames collected during an episode.

    Args:
        frames: A list of dictionaries containing the animation frames.
        filename: The output filename for the GIF animation.
        fps: The frames per second of the animation (default: 100).
    """
    images = [Image.fromarray(frame["frame"]) for frame in frames]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
    )

# Example usage
create_gif(frames, "animation.gif") #saves the GIF locally
