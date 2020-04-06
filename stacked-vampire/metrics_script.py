import os
import json
import pandas as pd

metrics_dir = "model_logs/clf/attn_ensemble_relu_weight_sum_0"

classes = ["Cancer", "Other NCD", "Diabetes", "Renal", "Stroke", "Liver",
           "Cardio", "Other Comm", "Pneumonia", "TB/AIDS", "Maternal", "External"]

metrics_dict = {}
split_precision = []
split_recall = []
split_f1 = []
metrics_file = metrics_dir + "/metrics.json"
if os.path.isfile(metrics_file):
    f = open(metrics_file)
    json_obj = json.load(f)

    for cl in classes:
        if 'best_validation_' + cl + '_P' in json_obj:
            precision = json_obj['best_validation_' + cl + '_P']
        else:
            precision = 0.0

        if 'best_validation_' + cl + '_R' in json_obj:
            recall = json_obj['best_validation_' + cl + '_R']
        else:
            recall = 0.0

        if 'best_validation_' + cl + '_F1' in json_obj:
            f1 = json_obj['best_validation_' + cl + '_F1']
        else:
            f1 = 0.0

        split_precision.append(precision)
        split_recall.append(recall)
        split_f1.append(f1)

    metrics_dict['precision'] = split_precision
    metrics_dict['recall'] = split_recall
    metrics_dict['f1'] = split_f1

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(metrics_dir + '/metrics.csv', index=False)