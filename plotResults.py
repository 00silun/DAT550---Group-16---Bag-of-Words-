import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Load per-sample predictions
predictions = pd.read_csv("word2vec_max_eval_predictions.csv")

true_labels = predictions['true_label_name']
predicted_labels = predictions['predicted_label_name']

# Get the sorted list of all labels
all_labels = sorted(predictions['true_label_name'].unique())

# Plot Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels, labels=all_labels)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=all_labels, yticklabels=all_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix (Labels)')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()
