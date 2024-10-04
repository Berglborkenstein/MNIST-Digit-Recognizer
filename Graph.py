# Graphs the improvement of the AI
import matplotlib.pyplot as plt

x1_values = []
x2_values = []
y_values = []

def graph(train_accuracy,test_accuracy,iteration_no,line1,line2,ax):

    x1_values.append(train_accuracy)
    x2_values.append(test_accuracy)
    y_values.append(iteration_no)

    line1.set_data(y_values,x1_values)
    line2.set_data(y_values,x2_values)

    ax.relim()
    ax.autoscale_view()

    plt.draw()
    plt.pause(0.01)