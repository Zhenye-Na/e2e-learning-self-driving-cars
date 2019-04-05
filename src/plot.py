"""PLot training and validation loss."""


import numpy as np
import matplotlib.pyplot as plt


training_loss = None
validation_loss = None

with open("loss2.txt") as f:
    for line_number, line in enumerate(f.readlines()):
        if line_number == 0:
            training_loss = line.rstrip().split(", ")
        else:
            validation_loss = line.rstrip().split(", ")

# print(training_loss)

training_loss = [float(loss) for loss in training_loss]
validation_loss = [float(loss) for loss in validation_loss]

print(len(training_loss))
print(len(validation_loss))

training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)

# print(training_loss)

plt.title("Training Loss vs Validation Loss")
plt.xlabel("Iterations")
plt.plot(range(len(training_loss)), training_loss, 'b', label='Training Loss')
plt.plot(np.linspace(0, len(training_loss), len(validation_loss)), validation_loss, 'g-.', label='Validation Loss')
plt.legend()

plt.show()
