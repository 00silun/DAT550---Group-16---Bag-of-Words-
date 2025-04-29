import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(
    model,
    X_test=None,
    y_test=None,
    test_loader=None,
    device='cpu',
    csv_filename='evaluation_results.csv',
    idx_to_label=None  
):
    """
    Evaluate a PyTorch model and save accuracy, precision, recall, F1-score
    AND individual predictions (numeric and label name) to CSV.
    """
    model.eval()
    all_preds = []  
    all_labels = []  

    with torch.no_grad():
        if test_loader:
            data_iter = test_loader
        elif X_test is not None and y_test is not None:
            data_iter = [(X_test, y_test)]
        else:
            raise ValueError("Provide either test_loader or X_test and y_test.")

        for X_batch, y_batch in data_iter:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    # Save metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro'),
        'recall': recall_score(all_labels, all_preds, average='macro'),
        'f1_score': f1_score(all_labels, all_preds, average='macro')
    }
    pd.DataFrame([metrics]).to_csv(csv_filename, index=False)

    # Save detailed predictions
    pred_data = {
        'true_label_idx': all_labels,
        'predicted_label_idx': all_preds,
    }

    if idx_to_label is not None:
        pred_data['true_label_name'] = [idx_to_label[idx] for idx in all_labels]
        pred_data['predicted_label_name'] = [idx_to_label[idx] for idx in all_preds]

    predictions_df = pd.DataFrame(pred_data)
    pred_csv_filename = csv_filename.replace(".csv", "_predictions.csv")
    predictions_df.to_csv(pred_csv_filename, index=False)

    print(f"Saved metrics to {csv_filename}")
    print(f"Saved predictions to {pred_csv_filename}")

    return metrics
