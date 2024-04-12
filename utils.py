import pandas as pd
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_SV(agent,height=15,width=20,pipe_gap=4,title='Q-learning'):

    def get_Z(x, y):
        if (x,y) in agent.state_index.keys():
            return np.max(agent.q[agent.state_index[(x,y)]])
        else:
            return 0

    def get_figure(ax):
        x_range=np.arange(0, width-int(width*0.3)-1)
        ymax= height-1-int(pipe_gap//2)-1
        y_range=np.arange(-ymax, ymax)

        X, Y = np.meshgrid(x_range, y_range)
        
        Z = np.array([get_Z(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=plt.cm.coolwarm)
        ax.set_xlabel('Distance x from the center of the pipe')
        ax.set_ylabel('Distance y from the center of the pipe')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title(title)
    get_figure(ax)
    plt.show()

def plot_policy(agent,height=15,width=20,pipe_gap=4,title='Q-learning'):
    def get_Z(x, y):
        if (x,y) in agent.state_index.keys():
            return np.argmax(agent.q[agent.state_index[(x,y)]])
        else:
            return -1

    def get_figure(ax):
        x_range=np.arange(0, width-int(width*0.3)-1)
        ymax= height-1-int(pipe_gap//2)-1
        y_range=np.arange(-ymax, ymax)

        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x,y) for x in x_range] for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2), vmin=0, vmax=1, extent=[0, width-int(width*0.3)-1, -ymax, ymax])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_xaxis()
        ax.set_xlabel('Distance x from the center of the pipe')
        ax.set_ylabel('Distance y from the center of the pipe')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[-1,0,1], cax=cax)
        cbar.ax.set_yticklabels(['unseen','0:Idle', '1:Flap'])
            
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title(title + ': Policy')
    get_figure(ax)
    plt.show()


def plot_results(all_run_rewards, all_run_scores,epsilon,step_size,discount,num_episodes):
    plt.figure(figsize=(14,5))
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    # Plot reward
    axs[0, 0].plot(all_run_rewards)
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')

    # Plot smoothed reward
    smoothed_reward = pd.Series(all_run_rewards).rolling(500, min_periods=1).mean()
    axs[0, 1].plot(smoothed_reward)
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Smoothed Reward')

    # Plot score
    axs[1, 0].plot(all_run_scores)
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Score')

    # Plot smoothed score
    smoothed_score = pd.Series(all_run_scores).rolling(500, min_periods=1).mean()
    axs[1, 1].plot(smoothed_score)
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Smoothed Score')

    plt.suptitle('Results for epsilon={}, step_size={}, discount={}, num_episodes={}'.format(epsilon,step_size,discount,num_episodes), fontsize=11)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

def plot_results_para(results):
    plt.figure(figsize=(17,6))
    cmap = plt.get_cmap('viridis')
    for i, (params, (rewards, scores, time)) in enumerate(results.items()):
        epsilon, step_size, discount = params
        smoothed_scores = pd.Series(scores).rolling(500, min_periods=1).mean()
        n=len(results.keys())

        color = cmap((i % n) / n)
        if discount == 1:
            plt.plot(smoothed_scores, linestyle='--', label=f"Epsilon={epsilon}, Step Size={step_size}, Discount={discount}")
        else:
            plt.plot(smoothed_scores, color=color, label=f"Epsilon={epsilon}, Step Size={step_size}, Discount={discount}")
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Training Results')
    plt.show()

def plot_time(results):
    time_values = []
    max_scores = []
    labels = []
    for i, (params, (_, scores, time)) in enumerate(results.items()):
        time_values.append(time/60)  
        max_scores.append(max(scores))
        labels.append(f'{params}')

    plt.figure(figsize=(15, 6))
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    ax1.plot(time_values, 'o-', color='green', label='Time')
    ax1.set_ylabel('Time (minutes)')

    ax2.plot(max_scores, 'o-', color='gray', label='Max Score')
    ax2.set_ylabel('Max Score')

    plt.xlabel('Parameter Choice')

    ax1.set_xticks(range(len(time_values)))
    ax1.set_xticklabels(labels, rotation=45)

    plt.title('Time vs Parameter Choice')
    plt.tight_layout()
    plt.show()

def plot_comparison(R1, R2,name1,name2):
    plt.figure(figsize=(14,5))
    sR1 = pd.Series(R1).rolling(500, min_periods=1).mean()
    sR2 = pd.Series(R2).rolling(500, min_periods=1).mean()
    plt.plot(sR1,label=name1)
    plt.plot(sR2,label=name2)
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.title(f'Comparison between {name1} and {name2}',fontsize=11)
    plt.show()