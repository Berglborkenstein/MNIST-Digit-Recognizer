# Creates a screen for the user to give the number of hidden layers, the size of each, the learning rate and the number of iterations
# Once those are acquired, the user is then shown a canvas. 
# The ai is trained according to the inputs
# The user then draws on the canvas
# They are then asked to submit their drawing
# The drawing is converted into a format that the ai can interpret
# The ai is then passed the csv drawing and outputs what it thinks the digit is
import tkinter as tk
from tkinter import ttk
from Digit_Recogniser import grad_descent, X_train,Y_train


root = tk.Tk()
root.title("Hello")
root.geometry("500x400")


def create_layers():

    # Clears the frame of any previous entries
    for widget in layers_frame.winfo_children():
        widget.destroy()
        root.update_idletasks()

    num_layers = hidden_layer_entry.get()

    if num_layers.isdigit():
        global layers
        layers = []
        
        if int(num_layers) > 0:
            num_layers = int(num_layers)
                
            a = ttk.Label(master = layers_frame, text = 'Please enter the size of each layer:', font = 'Calibri 14')
            a.pack()

            for i in range(num_layers):
                q = ttk.Entry(master = layers_frame)
                q.pack(pady = 2)
                layers.append(q)
            
        submit_button = ttk.Button(master = root, text = 'Submit', command = pass_on)
        submit_button.pack()

    else:
        error = ttk.Label(master = layers_frame, text = 'Please enter an integer greater than 0')
        error.pack()

def valid_iterations():
    iterations_valid = False

    iterations = iterations_entry.get()
    if not iterations.isdigit():
        error_iterations = ttk.Label(master = iterations_error, text = 'Please enter an integer')
        error_iterations.pack()
    elif int(iterations) <= 0:
        error_int_iterations = ttk.Label(master = iterations_error, text = 'Please enter an integer greater than 0')
        error_int_iterations.pack()
    else:
        for widget in iterations_error.winfo_children():
            widget.destroy()
            root.update_idletasks()
        iterations_valid = True

    return iterations_valid

def valid_learning():
    learning_valid = False
    # Learning Rate  
    learning_rate = learning_entry.get()
    try:
        learning_rate = float(learning_rate)
    except ValueError:
        error_learning = ttk.Label(master = learning_error, text = 'Please enter a number')
        error_learning.pack()
        return False
    
    if learning_rate <= 0:
        error_neg_learning = ttk.Label(master = learning_error, text = 'The number you enter must be greater than 0')
        error_neg_learning.pack()
    else:
        for widget in learning_error.winfo_children():
            widget.destroy()
            root.update_idletasks()
        learning_valid = True
    
    return learning_valid

def valid_layers():
    layers_valid = True
    # Layers
    layers_values = [layer.get() for layer in layers]
    for item in layers_values:
        if not item.isdigit() or int(item) <= 0:
            layers_valid = False 
    if not layers_valid:
        error_layers = ttk.Label(master = layers_error, text = 'Please ensure all inputs are integers greater than 0')
        error_layers.pack()
    else:
        layers_valid = True
        for widget in layers_error.winfo_children():
            widget.destroy()
        
    return layers_valid

def verify():

    iterations_valid = valid_iterations()
    
    learning_valid = valid_learning()

    layers_valid = valid_layers()

    if layers_valid and learning_valid and iterations_valid:
        return True
    else:
        return False


def pass_on():

    if verify():
        iterations = int(iterations_entry.get())
        learning_rate = float(learning_entry.get())
        hidden_size = [784] + [int(layer.get()) for layer in layers] + [10]
        grad_descent(X_train,Y_train,iterations,learning_rate,hidden_size)


# Initialises the hidden layer frame 
hidden_layer_frame = ttk.Frame(master = root)
hidden_layer_label = ttk.Label(master = hidden_layer_frame, text = 'Please enter the number of hidden layers:', font = 'Calibri 20')
hidden_layer_pair = ttk.Frame(master = hidden_layer_frame)

# Initialises the hidden layer entry
hidden_layer_entry = ttk.Entry(master = hidden_layer_pair)
hidden_layer_button = ttk.Button(master = hidden_layer_pair, text = 'Submit', command = create_layers)

layers_frame = ttk.Frame(master = hidden_layer_frame)
layers_error = ttk.Frame(master = layers_frame)

# Packs them accordingly
hidden_layer_frame.pack()
hidden_layer_label.pack()
hidden_layer_pair.pack(pady = 5)
hidden_layer_entry.pack(side = 'left')
hidden_layer_button.pack(side = 'left')
layers_frame.pack()
layers_error.pack()


# Initialises the iterations frame and entry
iterations_frame = ttk.Frame(master = root)
iterations_label = ttk.Label(master = iterations_frame, text = 'Please enter the number of iterations:', font = 'Calibri 20')
iterations_entry = ttk.Entry(master = iterations_frame)
iterations_error = ttk.Frame(master = iterations_frame)

# Packs the iterations frame
iterations_frame.pack()
iterations_label.pack()
iterations_entry.pack(pady=5)
iterations_error.pack()


# Initialises the learning rate frame and entry
learning_frame = ttk.Frame(master = root)
learning_label = ttk.Label(master = learning_frame, text = 'Please enter the learning rate of the AI:', font = 'Calibri 20')
learning_entry = ttk.Entry(master = learning_frame)
learning_error = ttk.Frame(master = learning_frame)

# Packs the learning rate frame
learning_frame.pack()
learning_label.pack()
learning_entry.pack(pady=5)
learning_error.pack()

root.mainloop()