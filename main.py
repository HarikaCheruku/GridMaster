import gym
import numpy as np
import random
import time

def train_q_learning(episodes=10000, learning_rate_start=0.1, gamma=0.99, epsilon_start=1.0):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    rewards_per_episode = []
    
    epsilon_decay = 0.995  
    epsilon_min = 0.01
    epsilon = epsilon_start
    
    learning_rate_decay = 0.999  
    learning_rate = learning_rate_start
    
    for episode in range(episodes):
        state = env.reset()[0]
        done = False
        total_reward = 0
        step_count = 0
        max_steps = 200  

        while not done and step_count < max_steps:
            
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  
            else:
                action = np.argmax(q_table[state, :])  
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            
            if terminated and reward == 0:  
                custom_reward = -10   
            elif terminated and reward == 1:  
                custom_reward = 100   
            else:  
                custom_reward = -1    
            
            
            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state, :])
            new_value = (1 - learning_rate) * old_value + learning_rate * (custom_reward + gamma * next_max)
            q_table[state, action] = new_value
            
            state = next_state
            total_reward += custom_reward  
            step_count += 1
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        learning_rate = max(0.01, learning_rate * learning_rate_decay)
        rewards_per_episode.append(total_reward)
        
        
        if (episode + 1) % 1000 == 0:
            avg_reward = np.mean(rewards_per_episode[-1000:])
            print(f"Episode {episode + 1}, Average Reward (last 1000): {avg_reward:.2f}")
    
    print("\nFinal Q-table:")
    print(q_table)
    
    np.save('Qtable.npy', q_table)
    print("\nQ-table saved to 'q_table.npy'")
    
    return q_table, rewards_per_episode

def demonstrate_optimal_path(q_table):
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='human')
    state = env.reset()[0]
    done = False
    step = 0
    max_steps = 200
    gamma = 0.99
    
    print("\nDemonstrating optimal path:")
    env.render()
    time.sleep(1)
    
    total_reward = 0
    
    while not done and step < max_steps:
        current_value = np.max(q_table[state])
        action = np.argmax(q_table[state, :])
        old_state = state
        
        print("\nQ-values for current state:")
        for act, q_val in enumerate(['LEFT', 'DOWN', 'RIGHT', 'UP']):
            print(f"{q_val}: {q_table[state][act]:.3f}")
        
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        
        if terminated and reward == 0:
            custom_reward = -10
        elif terminated and reward == 1:
            custom_reward = 100
        else:
            custom_reward = -1
            
        total_reward += custom_reward
        
        
        future_value = np.max(q_table[state]) * gamma if not done else custom_reward
        
        print(f"\nStep {step + 1}:")
        print(f"From state {old_state} to {state}")
        print(f"Action taken: {['LEFT', 'DOWN', 'RIGHT', 'UP'][action]}")
        print(f"Current state Q-value: {current_value:.3f}")
        print(f"Immediate reward: {custom_reward}")  
        print(f"Discounted future value: {future_value:.3f}")
        print(f"Total reward so far: {total_reward}")  
        
        env.render()
        time.sleep(0.5)
        step += 1
        
        if done:
            if custom_reward == 100:
                print(f"\nGoal reached in {step} steps!")
                print(f"Final total reward: {total_reward}")
            else:
                print("\nFell in a hole!")
                print(f"Final total reward: {total_reward}")
    
    if step == max_steps:
        print("Maximum steps reached without finding the goal.")
        print(f"Final total reward: {total_reward}")
    
    env.close()

def main():
    print("Starting training...")
    q_table, rewards = train_q_learning(episodes=10000)
    
    
    demonstrate_optimal_path(q_table)

if __name__ == "__main__":
    main()