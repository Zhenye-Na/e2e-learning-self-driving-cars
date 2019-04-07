"""
Plot training and validation loss.

@author: Zhenye Na - https://github.com/Zhenye-Na
@reference: "End to End Learning for Self-Driving Cars", arXiv:1604.07316
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns; sns.set(color_codes=True)

sns.set_style("whitegrid")

training_loss = []
validation_loss = []

with open("loss1.txt") as f:
    for line in f.readlines():
        if "Training" in line:
            training_loss.append(float(line.rstrip().split(": ")[-1]))
        elif "Validation" in line:
            validation_loss.append(float(line.rstrip().split(": ")[-1]))


training_loss = [float(loss) for loss in training_loss]
validation_loss = [float(loss) for loss in validation_loss]

training_loss = np.array(training_loss)
validation_loss = np.array(validation_loss)

# ax = sns.lineplot(x=np.linspace(0, len(training_loss), len(training_loss)), y=training_loss)
# ax = sns.lineplot(x=np.linspace(0, len(training_loss), len(validation_loss)), y=validation_loss)

plt.title("Training Loss vs Validation Loss")
plt.xlabel("Iterations")
plt.plot(range(len(training_loss)), training_loss, 'b', label='Training Loss')
plt.plot(np.linspace(0, len(training_loss), len(validation_loss)), validation_loss, 'g-.', label='Validation Loss')
plt.legend()
plt.grid(False)
plt.show()
