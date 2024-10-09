from tkinter import *
import numpy as np
from PIL import ImageGrab
from Digit_Recogniser import Forward_Prop
from shared_params import get_params
from tkinter.ttk import Progressbar

def canvas():
    window = Tk()
    window.title("Handwritten Digit Recognition")

    W,b,dropout = get_params()

    # Create a list to hold the progress bars and labels
    progress_bars = []
    labels = []
    for i in range(10):
        # Create a progress bar
        pb = Progressbar(window, orient=HORIZONTAL, length=200, mode='determinate')
        pb.place(x=500, y=70 + i * 30)  # Positioning each progress bar next to the canvas
        progress_bars.append(pb)

        # Create a label for each progress bar
        lbl = Label(window, text=f'{i}:', font=('Arial', 10))
        lbl.place(x=480, y=70 + i * 30)  # Position the label slightly to the left of the progress bar
        labels.append(lbl)

    def MyProject():
        global l1

        widget = cv
        # Setting co-ordinates of canvas
        x = window.winfo_rootx() + widget.winfo_x()
        y = window.winfo_rooty() + widget.winfo_y()
        x1 = x + widget.winfo_width()
        y1 = y + widget.winfo_height()

        # Image is captured from canvas and is resized to (28 X 28) px
        img = ImageGrab.grab().crop((x, y, x1, y1)).resize((28, 28))

        # Converting rgb to grayscale image
        img = img.convert('L')

        # Extracting pixel matrix of image and converting it to a vector of (1, 784)
        x = np.asarray(img)
        vec = np.zeros((784, 1))
        k = 0
        for i in range(28):
            for j in range(28):
                vec[k][0] = x[i][j]
                k += 1

        # Calling function for prediction
        X = vec / 255
        _,A,_ = Forward_Prop(X,W,b,dropout)
        estimate = [item[0] for item in A[-1]]
        
        # Update progress bars with predictions 
        for i in range(10):
            progress_bars[i]['value'] = estimate[i] * 100 # Assuming pred[i] is between 0 and 1

        window.after(100,lambda: MyProject())

    lastx, lasty = None, None

    # Clears the canvas and progress bars
    def clear_widget():
        cv.delete("all")

    # Activate canvas
    def event_activation(event):
        global lastx, lasty
        cv.bind('<B1-Motion>', draw_lines)
        lastx, lasty = event.x, event.y

    # To draw on canvas
    def draw_lines(event):
        global lastx, lasty
        x, y = event.x, event.y
        cv.create_line((lastx, lasty, x, y), width=30, fill='white', capstyle=ROUND, smooth=TRUE, splinesteps=12)
        lastx, lasty = x, y

    # Label
    L1 = Label(window, text="Handwritten Digit Recognition", font=('Algerian', 25), fg="blue")
    L1.place(x=70, y=10)

    # Button to clear canvas
    b1 = Button(window, text="Clear Canvas", font=('Algerian', 15), bg="orange", fg="black", command=clear_widget)
    b1.place(x=215, y=370)

    # Setting properties of canvas
    cv = Canvas(window, width=350, height=290, bg='black')
    cv.place(x=120, y=70)

    cv.bind('<Button-1>', event_activation)
    window.geometry("800x500")  # Adjust window width to accommodate progress bars

    MyProject()
    window.mainloop()
