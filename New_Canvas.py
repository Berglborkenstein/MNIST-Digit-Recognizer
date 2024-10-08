import pygame
import numpy as np
import cv2
import sys
from Digit_Recogniser import Forward_Prop, grad_descent, X_train, Y_train

# Constants
DRAW_RESOLUTION = 1024
OUTPUT_RESOLUTION = 28
BRUSH_RADIUS = 20
NUM_BARS = 10  # Number of progress bars
BAR_SPACING = 5  # Space between bars
FPS = 60  # Frame rate
DOWNSCALE_INTERVAL = 0.1  # Interval in seconds for downscaling (100ms)

# Initialize Pygame
pygame.init()

# Set up display
screen = pygame.display.set_mode((DRAW_RESOLUTION, DRAW_RESOLUTION))
pygame.display.set_caption("Drawing Canvas")

# Set up canvas
canvas_surface = pygame.Surface((DRAW_RESOLUTION, DRAW_RESOLUTION))
canvas_surface.fill((0, 0, 0))  # Start with a black canvas

# Progress bar initialization
progress_values = [0] * NUM_BARS  # Initialize progress values for each bar

# Font for labels
font = pygame.font.Font(None, 36)  # Default font and size

# Main loop
running = True
clock = pygame.time.Clock()

# Time tracker for controlling the downscaling frequency
last_downscale_time = 0

def draw_brush(surface, pos, radius, mode):
    x, y = pos
    for i in range(-radius, radius):
        for j in range(-radius, radius):
            # Ensure we're within canvas bounds
            if 0 <= x + i < DRAW_RESOLUTION and 0 <= y + j < DRAW_RESOLUTION:
                distance = np.sqrt(i**2 + j**2)
                if distance <= radius:
                    color = (255, 255, 255) if mode == 0 else (0, 0, 0)  # Draw or erase mode
                    surface.set_at((x + i, y + j), color)

def downscale_canvas(pygame_canvas, output_resolution=28):
    # Convert Pygame canvas (surface) to a string buffer
    canvas_str = pygame.image.tostring(pygame_canvas, "RGB")

    # Get the dimensions of the Pygame canvas
    width, height = pygame_canvas.get_size()

    # Convert the string buffer to a NumPy array and reshape to match canvas dimensions
    canvas_np = np.frombuffer(canvas_str, dtype=np.uint8).reshape((height, width, 3))

    # Convert the canvas to grayscale
    gray_canvas = cv2.cvtColor(canvas_np, cv2.COLOR_RGB2GRAY)

    # Resize the grayscale image to the target resolution
    downscaled = cv2.resize(gray_canvas, (output_resolution, output_resolution), interpolation=cv2.INTER_AREA)

    # Flatten the 28x28 image to a 784-element array
    return downscaled.flatten()

def draw_progress_bars(screen, values):
    bar_width = (DRAW_RESOLUTION - (NUM_BARS - 1) * BAR_SPACING) // NUM_BARS  # Width of each bar with spacing
    for i, value in enumerate(values):
        x_pos = i * (bar_width + BAR_SPACING)  # Position of the bar considering spacing
        pygame.draw.rect(screen, (255, 0, 0), (x_pos, DRAW_RESOLUTION - 50, bar_width, 20))  # Background
        pygame.draw.rect(screen, (0, 255, 0), (x_pos, DRAW_RESOLUTION - 50, int(bar_width * value), 20))  # Foreground
        
        # Draw label for the bar
        label_text = font.render(f"{i}", True, (255, 255, 255))  # Create label text
        label_rect = label_text.get_rect(center=(x_pos + bar_width // 2, DRAW_RESOLUTION - 75))  # Centered above the bar
        screen.blit(label_text, label_rect)  # Draw the label

def canvas_func(W, b):
    global last_downscale_time
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Clear canvas
                    canvas_surface.fill((0, 0, 0))

        mouse_buttons = pygame.mouse.get_pressed()
        if mouse_buttons[0] or mouse_buttons[2]:  # Left or right mouse button pressed
            mouse_pos = pygame.mouse.get_pos()
            mode = 0 if mouse_buttons[0] else 1  # Left click to draw, right click to erase

            # Draw brush stroke
            draw_brush(canvas_surface, mouse_pos, BRUSH_RADIUS, mode)

        # Downscale the canvas and update progress bars at regular intervals
        current_time = pygame.time.get_ticks() / 1000  # Get current time in seconds
        if current_time - last_downscale_time >= DOWNSCALE_INTERVAL:
            last_downscale_time = current_time
            downscaled_canvas = downscale_canvas(canvas_surface, OUTPUT_RESOLUTION)
            downscaled_canvas = downscaled_canvas.reshape(784, 1)  # Reshape for the neural network

            # Forward propagation through the neural network
            _, A, _ = Forward_Prop(downscaled_canvas, W, b)
            estimate = [item[0] for item in A[-1]]

            # Update the progress bar values
            for i in range(10):
                progress_values[i] = estimate[i]

        # Render canvas
        screen.fill((0, 0, 0))  # Clear the screen with black
        screen.blit(canvas_surface, (0, 0))
        
        # Draw progress bars
        draw_progress_bars(screen, progress_values)
        
        pygame.display.flip()
        clock.tick(FPS)  # Cap the frame rate

W, b = grad_descent(X_train, Y_train, 100, 0.1, [784, 10, 10])

canvas_func(W, b)

# Quit Pygame
pygame.quit()
sys.exit()