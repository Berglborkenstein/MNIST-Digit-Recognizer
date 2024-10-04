import tkinter as tk
from tkinter import ttk
import numpy as np
from Digit_Recogniser import Forward_Prop, grad_descent, X_train, Y_train
from PIL import Image

    # Canvas needs to have 28x28 'pixels'
    # Needs to have a paint and erase
    # Needs to be set up that only 
def canvas(W,b,parent):
    pixel_size = 20
    canvas_size = pixel_size * 28

    canvas_window = tk.Toplevel(parent)
    canvas_window.title('Canvas')
    canvas_window.geometry(f'{canvas_size + 130}x{canvas_size + 60}')

    def draw(event):
        x = event.x // pixel_size
        y = event.y // pixel_size

        for rect,rect_x,rect_y in pixels:
            if rect_x == x and rect_y == y:
                update_colour(rect,255)
            #elif (x,y) in [(rect_x + 1, rect_y), (rect_x - 1, rect_y), (rect_x, rect_y + 1), (rect_x, rect_y - 1)]:
             #   update_colour(rect,30)
            #elif (x,y) in [(rect_x + 1, rect_y + 1), (rect_x - 1, rect_y - 1), (rect_x - 1, rect_y + 1), (rect_x + 1, rect_y - 1)]:
             #   update_colour(rect,20)
        
    def update_colour(rect,change):
        c = digit_canvas.itemcget(rect,'fill')
        current_colour = hex_to_rgb(c)[0]
        current_colour = max(current_colour - change, 0)
        digit_canvas.itemconfig(rect, fill = rgb_to_hex((current_colour,) * 3), outline = rgb_to_hex((current_colour,) * 3))

    def erase(event):
        x = (event.x // pixel_size) 
        y = (event.y // pixel_size) 

        white = rgb_to_hex((255,)*3)
        for rect,rect_x,rect_y in pixels:
            if rect_x == x and rect_y == y:
                digit_canvas.itemconfig(rect, fill = white, outline = white)

    def clear():
        for item,_,_ in pixels:
            white = rgb_to_hex((255,)*3)
            digit_canvas.itemconfig(item, fill = white, outline = white)

    def pass_to_ai(W,b):
        X = canvas_to_csv()
        _,A = Forward_Prop(X,W,b)
        estimate = [item[0] for item in A[-1]]

        for i in range(10):
            confidences[i].set(estimate[i])

        canvas_window.after(100,lambda: pass_to_ai(W,b))

    def canvas_to_csv():
        grayscale_canvas = [(1 - hex_to_rgb(digit_canvas.itemcget(item,'fill'))[0] / 255) for item,_,_ in pixels]
        #display_test_images(np.array(grayscale_canvas))
        x = (np.array(grayscale_canvas) * 255)
        #img = Image.fromarray(x.reshape((28,28)).astype(np.uint8),mode = 'L')
        #img.show()
        return np.array(grayscale_canvas).reshape(784, 1)

    def rgb_to_hex(rgb):
        r, g, b = rgb
        return f'#{r:02x}{g:02x}{b:02x}'

    def hex_to_rgb(hex_color):
        # Remove the '#' character if it's there
        hex_color = hex_color.lstrip('#')
        # Convert the hex string to RGB values
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        
        return (r, g, b)

    top = ttk.Frame(master = canvas_window)
    top.pack()

    digit_canvas = tk.Canvas(master = top, bg = 'white', width = canvas_size, height = canvas_size)
    digit_canvas.pack(side = 'left')

    # Draw the initial grid
    pixels = []
    for x in range(0, canvas_size, pixel_size):
        for y in range(0, canvas_size, pixel_size):
            pixels.append((digit_canvas.create_rectangle(y, x, y + pixel_size, x + pixel_size, outline=rgb_to_hex((255,)*3), fill=rgb_to_hex((255,)*3)),y/pixel_size,x/pixel_size))

    # Bind mouse click and drag to draw
    digit_canvas.bind("<B1-Motion>", draw)
    digit_canvas.bind("<Button-1>", draw)

    digit_canvas.bind("<B3-Motion>",erase)
    digit_canvas.bind("<Button-3>", erase)

    # Shows sliders of what the AI believes the answer to be
    confidence_frame = ttk.Frame(master = top)
    confidences = [tk.DoubleVar(value=0) for _ in range(10)]
    frames = [ttk.Frame(master = confidence_frame) for i in range(10)]
    confidence_bars = [ttk.Progressbar(master = frames[i], variable = confidences[i], maximum = 1, length = 100) for i in range(10)]
    labels = [ttk.Label(master = frames[i], text = f'{i}:') for i in range(10)]

    confidence_frame.pack(side = 'left', padx = 5)

    for i in range(10):
        confidence = confidence_bars[i]
        label = labels[i]
        frame = frames[i]

        frame.pack()
        label.pack(pady = 5, side = 'left')
        confidence.pack(pady = 5, padx = 3, side = 'left')

    # Initialises and packs the info and clear button at the bottom of the screen
    bottom_frame = ttk.Frame(master = canvas_window)
    info_label = ttk.Label(master = bottom_frame, text = 'Right-Click to Erase')
    info_label_2 = ttk.Label(master = bottom_frame, text = 'Left-Click to Draw')
    clear_button = ttk.Button(master = canvas_window, text = 'Clear Canvas', command = clear)

    bottom_frame.pack(pady = 5)
    info_label.pack(side = 'left')
    info_label_2.pack(side = 'left', padx = 10)
    clear_button.pack()

    pass_to_ai(W,b)

#W,b = grad_descent(X_train,Y_train,100,0.1,[784,100,10])
#canvas(W,b)