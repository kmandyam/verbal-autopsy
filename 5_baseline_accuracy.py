import pandas as pd
import re


directory = 'baseline_predictions_dar'
splits = ['splitA', 'splitB', 'splitC']
methods = ['tariff2', 'interva2', 'nbc2', 'insilico2']

classes = ["Cancer", "Other NCD", "Diabetes", "Renal", "Stroke", "Liver",
           "Cardio", "Other Comm", "Pneumonia", "TB/AIDS", "Maternal", "External"]

def f1_class(label, AP_true, AP_pred):
    tp = 0
    fp = 0
    fn = 0

    for i in range(len(AP_pred)):
        if AP_true[i] == label:
            if AP_pred[i] == label:
                tp += 1
            else:
                fn += 1
        elif AP_true[i] != label and AP_pred[i] == label:
            fp += 1

    precision = 0
    if tp + fp != 0:
        precision = tp / (tp + fp)
    recall = 0
    if tp + fn != 0:
        recall = tp / (tp + fn)
    f1 = 0
    if precision + recall != 0:
        f1 = (2 * precision * recall) / (precision + recall)
    return precision, recall, f1

# get csv split by split
metrics_dict = {}
for split in splits:
    split_precision = []
    split_recall = []
    split_f1 = []

    for method in methods:
        print(method + " " + split)

        predictions_path = directory + '/' + split + '/testing_dar/' + method + '.txt'
        gold_path = directory + '/' + split + '/' + 'test.csv'
        gold_df = pd.read_csv(gold_path)

        gold_labels = []
        for index, row in gold_df.iterrows():
            gold_labels.append(row['gs_text34'])

        predictions = []
        for line in open(predictions_path):
            pred = re.split(r'\t', line.rstrip())
            cod = pred[1].replace("\"", "")
            predictions.append(cod)

        assert len(predictions) == len(gold_labels)

        for cl in classes:
            precision, recall, f1 = f1_class(cl, gold_labels, predictions)

            split_precision.append(precision)
            split_recall.append(recall)
            split_f1.append(f1)

            # print stuff to verify
            print("Class: ", cl)
            print("Precision: ", precision)
            print("Recall: ", recall)
            print("F1: ", f1)
            print()

        print()
        print()
        print()

    metrics_dict[split + '_precision'] = split_precision
    metrics_dict[split + '_recall'] = split_recall
    metrics_dict[split + '_f1'] = split_f1

metrics_df = pd.DataFrame(metrics_dict)
metrics_df.to_csv(directory + '/' + 'metrics.csv', index=False)