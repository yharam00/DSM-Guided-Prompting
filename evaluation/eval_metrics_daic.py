import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix

file_path = 'data/eval/predict_label_daic.xlsx'
df = pd.read_excel(file_path)

print('Columns in the dataset:', df.columns.tolist())
print('Original data:')
print(df.head())

predict_cols = ['predict1', 'predict2', 'predict3', 'predict4', 'predict5', 'predict6', 'predict7', 'predict8', 'predict9']

label_mapping = {'Non-depressed.': 0, 'Non-depressed': 0, 'non-depressed': 0, 'depressed': 1, 'Depressed': 1}

for col in predict_cols:
    if col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].map(label_mapping).fillna(df[col])
        df[col] = df[col].astype(int)

print('\nAfter coding (0=non-depressed, 1=depressed):')
print(df.head())

true_col = 'label'
model_cols = [col for col in df.columns if col != true_col]

y_true = df[true_col]

results = []
for model in model_cols:
    y_pred = df[model]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    sens = recall_score(y_true, y_pred, pos_label=1)
    spec = recall_score(y_true, y_pred, pos_label=0)

    f1_per_class = f1_score(y_true, y_pred, average=None)
    f1_non_dep = f1_per_class[0]
    f1_dep = f1_per_class[1]
    f1_macro = f1_score(y_true, y_pred, average='macro')

    results.append({
        'Model': model,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
        'Accuracy': acc,
        'Mean F1': f1_macro
    })

metrics_df = pd.DataFrame(results)

print('\nClassification Metrics for Each Model:')
print(metrics_df.to_string(index=False))
