
import baseline
import pybullet
import gym
import pybullet_envs
import time

from baselines import deepq
from baselines.ddpg.ddpg import DDPG

def callback(lcl, _glb):
    # stop training if reward exceeds 199
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 199
    return is_solved

def train(env, name,callback):
    model = deepq.models.mlp([100,20,20])

    act = DDPG.train(
            env,
            q_func=model,
            lr=1e-3,
            max_timesteps=100000,
            buffer_size=50000,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            print_freq=10,
            callback=callback
        )
    print("Saving model to "+name+".pkl")
    act.save(""+name+".pkl")



pybullet.connect(pybullet.DIRECT)
env = gym.make("HumanoidBulletEnv-v0")
train(env,'Humanoid_model',callback)

act = deepq.load("Humanoid_model.pkl")

env.render('human')
obs, done = env.reset(), False
for _ in range(10000):
    env.render('human')
    obs, rew, done, _ = env.step(act(obs[ None ])[ 0 ])


