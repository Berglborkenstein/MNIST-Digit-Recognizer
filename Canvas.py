import tkinter as tk
from tkinter import ttk
import numpy as np
from Digit_Recogniser import Forward_Prop, grad_descent, X_train, Y_train
from PIL import Image, ImageDraw, ImageOps

def canvas(W, b, parent, dropout):
    canvas_size = 560  # Set canvas size to 560x560 for a 20-pixel brush on a 28x28 logical grid
    brush_radius = 10  # Brush radius to give a more natural stroke

    # Create a PIL image to store canvas drawing
    pil_image = Image.new('L', (canvas_size, canvas_size), color=255)
    draw_image = ImageDraw.Draw(pil_image)

    canvas_window = tk.Toplevel(parent)
    canvas_window.title('Canvas')
    canvas_window.geometry(f'{canvas_size + 130}x{canvas_size + 60}')

    def draw(event):
        x, y = event.x, event.y

        # Draw on the Tkinter canvas
        digit_canvas.create_oval(x - brush_radius, y - brush_radius, x + brush_radius, y + brush_radius, 
                                 fill='black', outline='black')

        # Draw on the PIL image
        draw_image.ellipse([x - brush_radius, y - brush_radius, x + brush_radius, y + brush_radius], fill=0)

    def erase(event):
        x, y = event.x, event.y

        # Erase on the Tkinter canvas by drawing a white circle
        digit_canvas.create_oval(x - brush_radius, y - brush_radius, x + brush_radius, y + brush_radius, 
                                 fill='white', outline='white')

        # Erase on the PIL image
        draw_image.ellipse([x - brush_radius, y - brush_radius, x + brush_radius, y + brush_radius], fill=255)

    def clear():
        # Clear the Tkinter canvas
        digit_canvas.delete("all")

        # Clear the PIL image to pure white
        draw_image.rectangle([0, 0, canvas_size, canvas_size], fill=255)

    def pass_to_ai(W, b):
        # Convert the PIL image to 28x28, invert it, and format it as an array
        resized_image = pil_image.resize((28, 28))
        inverted_image = ImageOps.invert(resized_image)

        # Convert to numpy array to find the bounding box of the drawn digit
        np_image = np.array(inverted_image)
        non_empty_columns = np.where(np_image.min(axis=0) < 255)[0]
        non_empty_rows = np.where(np_image.min(axis=1) < 255)[0]

        if non_empty_columns.size and non_empty_rows.size:
            # Calculate the bounding box
            crop_box = (min(non_empty_columns), min(non_empty_rows),
                        max(non_empty_columns), max(non_empty_rows))
            
            # Crop the image to the bounding box
            cropped_image = inverted_image.crop(crop_box)

            # Create a new 28x28 image and paste the cropped image at the center
            centered_image = Image.new('L', (28, 28), color=255)
            paste_x = (28 - cropped_image.width) // 2
            paste_y = (28 - cropped_image.height) // 2
            centered_image.paste(cropped_image, (paste_x, paste_y))

            # Convert the centered image to a grayscale array and normalize
            grayscale_array = np.array(centered_image).reshape(784, 1) / 255
        else:
            # If nothing is drawn, just use the empty image
            grayscale_array = np.array(inverted_image).reshape(784, 1) / 255


        _, A, _ = Forward_Prop(grayscale_array, W, b, dropout)
        estimate = [item[0] for item in A[-1]]

        for i in range(10):
            confidences[i].set(estimate[i])

        canvas_window.after(1000, lambda: pass_to_ai(W, b))

    top = ttk.Frame(master=canvas_window)
    top.pack()

    digit_canvas = tk.Canvas(master=top, bg='white', width=canvas_size, height=canvas_size)
    digit_canvas.pack(side='left')

    # Bind mouse click and drag to draw with a brush
    digit_canvas.bind("<B1-Motion>", draw)
    digit_canvas.bind("<Button-1>", draw)

    # Bind right-click to erase
    digit_canvas.bind("<B3-Motion>", erase)
    digit_canvas.bind("<Button-3>", erase)

    confidence_frame = ttk.Frame(master=top)
    confidences = [tk.DoubleVar(value=0) for _ in range(10)]
    frames = [ttk.Frame(master=confidence_frame) for i in range(10)]
    confidence_bars = [ttk.Progressbar(master=frames[i], variable=confidences[i], maximum=1, length=100) for i in range(10)]
    labels = [ttk.Label(master=frames[i], text=f'{i}:') for i in range(10)]

    confidence_frame.pack(side='left', padx=5)

    for i in range(10):
        frame = frames[i]
        label = labels[i]
        confidence = confidence_bars[i]

        frame.pack()
        label.pack(pady=5, side='left')
        confidence.pack(pady=5, padx=3, side='left')

    bottom_frame = ttk.Frame(master=canvas_window)
    info_label = ttk.Label(master=bottom_frame, text='Right-Click to Erase')
    info_label_2 = ttk.Label(master=bottom_frame, text='Left-Click to Draw')
    clear_button = ttk.Button(master=canvas_window, text='Clear Canvas', command=clear)

    bottom_frame.pack(pady=5)
    info_label.pack(side='left')
    info_label_2.pack(side='left', padx=10)
    clear_button.pack()

    pass_to_ai(W, b)
