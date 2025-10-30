#!/usr/bin/env python3
"""
Simple test to see if we can get the visual environment working.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

def test_environment():
    print('🎮 Testing environment visualization...')
    
    try:
        from env_platform import Platform2DEnv
        
        print('Creating environment with visual mode...')
        env = Platform2DEnv(render_mode="human", render_every=1, single_level=True)
        
        print('Environment created successfully!')
        print('Resetting environment...')
        obs, info = env.reset()
        
        print('Environment reset!')
        print('Taking a few random actions to test rendering...')
        
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            
            if terminated or truncated:
                print(f'Episode ended at step {i}')
                break
        
        print('Test completed!')
        env.close()
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_environment()
