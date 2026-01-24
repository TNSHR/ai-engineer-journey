


from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

y_true = [1,1,1,0,0,1,0]
y_pred = [1,0,1,0,1,1,0]

print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(classification_report(y_true, y_pred))