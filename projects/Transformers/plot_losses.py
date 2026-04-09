import json
import matplotlib.pyplot as plt

with open('projects/Transformers/losses.json', 'r') as f:
    history = json.load(f)

plt.figure(figsize=(10, 6))
plt.plot(history['step'], history['train'], label='Train Loss')
plt.plot(history['step'], history['val'], label='Val Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('NanoLanguageModel Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('projects/Transformers/loss_plot.png')
print("Plot saved to projects/Transformers/loss_plot.png")
