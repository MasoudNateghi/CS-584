import os
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from utils.variables import model_type

model_dir = f'misc/models/{model_type}'
metrics_path = f'{model_dir}/metrics.pkl'
with open(metrics_path, "rb") as f:
    metrics = pkl.load(f)
train_loss = metrics["train_loss"]
train_acc = metrics["train_acc"]
val_loss = metrics["val_loss"]
val_acc = metrics["val_acc"]

result_path = f'misc/results/{model_type}'
os.makedirs(result_path, exist_ok=True)

plt.figure()
plt.plot(np.arange(1, len(train_loss) + 1), train_loss, label="Train Loss")
plt.plot(np.arange(1, len(val_loss) + 1), val_loss, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig(f"{result_path}/loss.png")
plt.show()

plt.figure()
plt.plot(np.arange(1, len(train_acc) + 1), train_acc, label="Train Accuracy")
plt.plot(np.arange(1, len(val_acc) + 1), val_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.savefig(f"{result_path}/accuracy.png")
plt.show()