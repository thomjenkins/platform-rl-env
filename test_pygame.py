#!/usr/bin/env python3
"""
Simple PyGame test to see if windows are working.
"""

import os
os.environ['SDL_VIDEODRIVER'] = 'cocoa'

import pygame
import time

def test_pygame():
    print('🧪 Testing PyGame window creation...')
    
    try:
        pygame.init()
        print('✅ PyGame initialized')
        
        screen = pygame.display.set_mode((800, 600))
        print('✅ Display created')
        
        pygame.display.set_caption("PyGame Test Window")
        print('✅ Window title set')
        
        clock = pygame.time.Clock()
        
        print('🎮 Drawing test content...')
        
        for i in range(100):  # Run for 100 frames
            # Clear screen
            screen.fill((50, 50, 150))  # Dark blue
            
            # Draw a moving rectangle
            x = 50 + (i * 5) % 700
            y = 300
            pygame.draw.rect(screen, (255, 0, 0), (x, y, 50, 50))
            
            # Draw some text
            font = pygame.font.Font(None, 36)
            text = font.render(f"Frame {i}", True, (255, 255, 255))
            screen.blit(text, (10, 10))
            
            pygame.display.flip()
            clock.tick(60)
            
            # Check for quit
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print('👋 Window closed by user')
                    pygame.quit()
                    return
        
        print('✅ Test completed successfully!')
        pygame.quit()
        
    except Exception as e:
        print(f'❌ PyGame test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_pygame()
