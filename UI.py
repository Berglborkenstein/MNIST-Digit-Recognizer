import tkinter as tk
from tkinter import ttk
from Digit_Recogniser import grad_descent, X_train,Y_train
from Canvas import canvas


root = tk.Tk()
root.title("Hello")
root.geometry("500x400")


def create_layers():
        

    # Clears the frame of any previous entries
    for widget in layers_frame.winfo_children():
        widget.destroy()
        layers_frame.update_idletasks()
        
    for widget in submit_layer.winfo_children():
        widget.destroy()
        submit_layer.update_idletasks()

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

        submit_button = ttk.Button(master = submit_layer, text = 'Submit', command = pass_on)
        submit_button.pack()

    else:
        error = ttk.Label(master = layers_frame, text = 'Please enter an integer greater than 0')
        error.pack()

def valid_iterations():
    iterations_valid = False

    for widget in iterations_error.winfo_children():
        widget.destroy()
        iterations_error.update_idletasks()

    iterations = iterations_entry.get()
    if not iterations.isdigit():
        error_iterations = ttk.Label(master = iterations_error, text = 'Please enter an integer')
        error_iterations.pack()
    elif int(iterations) <= 0:
        error_int_iterations = ttk.Label(master = iterations_error, text = 'Please enter an integer greater than 0')
        error_int_iterations.pack()
    else:
        iterations_valid = True

    return iterations_valid

def valid_learning():
    learning_valid = False

    for widget in learning_error.winfo_children():
        widget.destroy()
    learning_error.update_idletasks()

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
        learning_valid = True
    
    return learning_valid


def valid_layers():
    layers_valid = True
    
    for widget in layers_error.winfo_children():
        widget.destroy()
    layers_error.update_idletasks()

    for layer in layers:
        layer_val = layer.get()

        try:
            layer_val = int(layer_val)

            if layer_val <= 0:
                error_int_layers = ttk.Label(master = layers_error, text = 'Please enter an integer greater than 0.')
                error_int_layers.pack()

                layers_valid = False
                break

        except ValueError:
            error_layers = ttk.Label(master = layers_error, text = 'Please enter an integer.')
            error_layers.pack()

            layers_valid = False
            break

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
        W, b = grad_descent(X_train,Y_train,iterations,learning_rate,hidden_size)
        canvas(W,b,root)


# Initialises the hidden layer frame 
hidden_layer_frame = ttk.Frame(master = root)
hidden_layer_label = ttk.Label(master = hidden_layer_frame, text = 'Please enter the number of hidden layers:', font = 'Calibri 20')
hidden_layer_pair = ttk.Frame(master = hidden_layer_frame)

# Initialises the hidden layer entry
hidden_layer_entry = ttk.Entry(master = hidden_layer_pair)
hidden_layer_button = ttk.Button(master = hidden_layer_pair, text = 'Submit', command = create_layers)

layers_frame = ttk.Frame(master = hidden_layer_frame)
layers_error = ttk.Frame(master = hidden_layer_frame)

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

submit_layer = ttk.Frame(master = root)
submit_layer.pack()

root.mainloop()

