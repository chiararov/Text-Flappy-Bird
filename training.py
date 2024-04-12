import time as tm
from agent import *
import gym
import numpy as np  
from tqdm import tqdm

def train(epsilon, step_size, discount, num_episodes=10000 ,num_runs=1,eps_decay=0.999,eps_min=1e-5,algorithm='ExpectedSarsa'):
    
    start_time=tm.time()

    env= gym.make('TextFlappyBird-v0', height = 15, width = 20, pipe_gap = 4)
    num_actions=env.action_space.n
    num_states= env.observation_space[0].n * env.observation_space[1].n
    seed=123
    agent_init_info = {"num_actions": num_actions, "num_states": num_states, "epsilon": epsilon, "step_size": step_size, "discount": discount, "seed": seed, "eps_decay": eps_decay, "eps_min": eps_min}


    all_run_rewards = []
    all_run_scores = []

    for _ in range(num_runs):
        if algorithm=='ExpectedSarsa':
            agent=ExpectedSarsaAgent()
        else:
            agent=QLearningAgent()
        agent.agent_init(agent_init_info)

        one_run_rewards=[]
        one_run_scores=[]

        for _ in tqdm(range(num_episodes)):
            obs = env.reset()
            state=obs[0]
            action = agent.agent_start(state)
            reward=0
            sum_reward=0
            # iterate
            while True:
                action = agent.agent_step(reward, state)
                obs, reward, done, _, info = env.step(action)
                state=obs
                sum_reward+=reward
                if done or info['score']>1000:
                    agent.agent_end(reward)
                    one_run_rewards.append(sum_reward)
                    one_run_scores.append(info['score'])
                    break

        all_run_rewards.append(np.array(one_run_rewards))
        all_run_scores.append(np.array(one_run_scores))
        env.close()
    total_time=tm.time()-start_time
    if num_runs==1:
        return one_run_rewards, one_run_scores,total_time,agent
    else:
        return all_run_rewards, all_run_scores,total_time,agent