from flask import Flask, render_template, Response, request
import gym
import numpy as np
from PIL import Image
import io
import base64
import time
import json
import random

app = Flask(__name__)


q_table = np.load('Qtable.npy')

env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode='rgb_array')
gamma = 0.99  

def reset_env():
    state = env.reset()[0]
    return state

def get_frame():
    frame = env.render()
    img = Image.fromarray(frame)
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    base64_img = base64.b64encode(img_io.read()).decode('utf-8')
    return base64_img

def get_custom_reward(terminated, reward):
    if terminated and reward == 0:
        return -10  
    elif terminated and reward == 1:
        return 100  
    return -1  

def run_agent_scenario(scenario_type='optimal'):
    state = reset_env()
    done = False
    step = 0
    max_steps = 100
    total_reward = 0
    
    while not done and step < max_steps:
        
        current_q_value = np.max(q_table[state, :])
        
        
        if scenario_type == 'optimal':
            action = np.argmax(q_table[state, :])
        elif scenario_type == 'suboptimal':
            if random.random() < 0.3:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])
        else:  
            if random.random() < 0.8:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_table[state, :])

        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        
        intermediate_reward = get_custom_reward(terminated, reward)
        total_reward += intermediate_reward
        
        
        if not done:
            discounted_value = gamma * np.max(q_table[next_state, :])
        else:
            discounted_value = intermediate_reward

        frame = get_frame()
        
        
        data = json.dumps({
            'frame': frame,
            'step': step + 1,
            'intermediate_reward': intermediate_reward,
            'total_reward': total_reward,
            'q_value': float(current_q_value),
            'discounted_value': float(discounted_value),
            'done': done,
            'scenario_type': scenario_type,
            'failed': done and reward == 0,
            'success': done and reward == 1
        })

        yield f"data: {data}\n\n"
        state = next_state
        step += 1
        time.sleep(0.5)

    
    if scenario_type == 'failure' and reward != 0 and step < max_steps:
        yield from run_agent_scenario('failure')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    scenario_type = request.args.get('type', 'failure')
    return Response(run_agent_scenario(scenario_type), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)