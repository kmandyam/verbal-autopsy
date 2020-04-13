import os
import re
import json
import pandas as pd
from tqdm import tqdm
# a script for generating data to train VAMPIRE (which includes the predictions of baseline classifiers)
# we need to generate three files: out-domain.jsonl, train.jsonl, and test.jsonl

# pre-work
labels = ["Cancer", "Other NCD", "Diabetes", "Renal", "Stroke", "Liver", "Cardio", "Other Comm",
          "Pneumonia", "TB/AIDS", "Maternal", "External"]

codebook_file = "raw/codebookfile.csv"
cb_df = pd.read_csv(codebook_file)

metadata_file = "raw/metadata.txt"

metadata = []
with open(metadata_file) as f:
    for line in f:
        metadata.append(line.strip())

output_dir = 'vampire_jsonl/splitA'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

def read_baseline(prediction_file):
    predictions = []
    with open(prediction_file) as f:
        for line in f.readlines():
            line = line.strip().split("\"")
            label = line[-2]
            assert label in labels or label == "Undetermined"
            predictions.append(label)
    return predictions


'''
Generating out-domain.jsonl
'''
print("Generating out-domain.jsonl....")
out_domain_jsonl = open(output_dir + "/out-domain.jsonl", "w", encoding='utf-8')
splits = ['split_1', 'split_2', 'split_3', 'split_4', 'split_5']

for split in splits:
    # read the baseline predictions from the output files
    tariff = read_baseline('augmented_dataset/splitA/' + split + '/tariff2.txt')
    interva = read_baseline('augmented_dataset/splitA/' + split + '/interva2.txt')
    nbc = read_baseline('augmented_dataset/splitA/' + split + '/nbc2.txt')
    insilico = read_baseline('augmented_dataset/splitA/' + split + '/insilico2.txt')

    # read the csv representing the portion of the dataset we have predictions for
    df = pd.read_csv('augmented_dataset/splitA/' + split + '/test.csv')

    # specify the excluded columns and bow columns
    bow_columns = df.filter(regex="word_*").columns
    non_use_cols = ["site", "module", "gs_code34", "gs_text34", "va34", "gs_code46",
                    "gs_text46", "va46", "gs_code55", "gs_text55", "va55", "gs_comorbid1",
                    "gs_comorbid2", "gs_level"]

    for index, row in tqdm(df.iterrows()):
        # gather text from one hot vector encoding columns
        text = []
        for col in bow_columns:
            if row[col] == 1:
                word = col[5:]
                text.append(word)

        metadata_text = []

        # if the column is in the metadata we provided at the top
        for col in df:
            if col not in bow_columns and col not in non_use_cols \
                    and col in metadata:
                # append the name of the column to the answer
                # if we have a name for the column
                col_name_df = cb_df[cb_df.variable == str(col)]
                if len(col_name_df) == 1:
                    col_name = col_name_df.iloc[0].question
                    try:
                        value = float(row[col])
                        value = str(int(value))
                    except ValueError:
                        value = str(row[col])
                        value = re.sub(r'\s+|\[|\]|\'', '', value)

                    covariate_name = re.sub(r'\s+|\[|\]|\'', '', col_name)
                    metadata_text.append(covariate_name + value)

        datum = {
            "label": row['gs_text34'],
            "text": ' '.join(text),
            "metadata_text": ' '.join(metadata_text),
            "site": row['site'],
            "tariff": tariff[index],
            "interva": interva[index],
            "nbc": nbc[index],
            "insilico": insilico[index]
        }

        # write all the outputs to the same jsonl file
        out_domain_jsonl.write(json.dumps(datum) + "\n")


'''
Generating train.jsonl
'''
print("Generating train.jsonl....")
train_jsonl = open(output_dir + "/train.jsonl", "w", encoding='utf-8')

# gather the predictions from the appropriate location
tariff = read_baseline('baseline_predictions_dar/splitA/training_dar/tariff2.txt')
interva = read_baseline('baseline_predictions_dar/splitA/training_dar/interva2.txt')
nbc = read_baseline('baseline_predictions_dar/splitA/training_dar/nbc2.txt')
insilico = read_baseline('baseline_predictions_dar/splitA/training_dar/insilico2.txt')

# open the csv that has the original data corresponding to the predictions
df = pd.read_csv('baseline_predictions_dar/splitA/train.csv')

assert len(df) == len(tariff)
assert len(df) == len(interva)
assert len(df) == len(nbc)
assert len(df) == len(insilico)

# specify the excluded columns and bow columns
bow_columns = df.filter(regex="word_*").columns
non_use_cols = ["site", "module", "gs_code34", "gs_text34", "va34", "gs_code46",
                "gs_text46", "va46", "gs_code55", "gs_text55", "va55", "gs_comorbid1",
                "gs_comorbid2", "gs_level"]

for index, row in tqdm(df.iterrows()):
    # gather text from one hot vector encoding columns
    text = []
    for col in bow_columns:
        if row[col] == 1:
            word = col[5:]
            text.append(word)

    metadata_text = []

    # if the column is in the metadata we provided at the top
    for col in df:
        if col not in bow_columns and col not in non_use_cols \
                and col in metadata:
            # append the name of the column to the answer
            # if we have a name for the column
            col_name_df = cb_df[cb_df.variable == str(col)]
            if len(col_name_df) == 1:
                col_name = col_name_df.iloc[0].question
                try:
                    value = float(row[col])
                    value = str(int(value))
                except ValueError:
                    value = str(row[col])
                    value = re.sub(r'\s+|\[|\]|\'', '', value)

                covariate_name = re.sub(r'\s+|\[|\]|\'', '', col_name)
                metadata_text.append(covariate_name + value)

    datum = {
        "label": row['gs_text34'],
        "text": ' '.join(text),
        "metadata_text": ' '.join(metadata_text),
        "site": row['site'],
        "tariff": tariff[index],
        "interva": interva[index],
        "nbc": nbc[index],
        "insilico": insilico[index]
    }

    # write all the outputs to the same jsonl file
    train_jsonl.write(json.dumps(datum) + "\n")


'''
Generating test.jsonl
'''
print("Generating test.jsonl....")
test_jsonl = open(output_dir + "/test.jsonl", "w", encoding='utf-8')

# gather the predictions from the appropriate location
tariff = read_baseline('baseline_predictions_dar/splitA/testing_dar/tariff2.txt')
interva = read_baseline('baseline_predictions_dar/splitA/testing_dar/interva2.txt')
nbc = read_baseline('baseline_predictions_dar/splitA/testing_dar/nbc2.txt')
insilico = read_baseline('baseline_predictions_dar/splitA/testing_dar/insilico2.txt')

# open the csv that has the original data corresponding to the predictions
df = pd.read_csv('baseline_predictions_dar/splitA/test.csv')

assert len(df) == len(tariff)
assert len(df) == len(interva)
assert len(df) == len(nbc)
assert len(df) == len(insilico)

# specify the excluded columns and bow columns
bow_columns = df.filter(regex="word_*").columns
non_use_cols = ["site", "module", "gs_code34", "gs_text34", "va34", "gs_code46",
                "gs_text46", "va46", "gs_code55", "gs_text55", "va55", "gs_comorbid1",
                "gs_comorbid2", "gs_level"]

for index, row in tqdm(df.iterrows()):
    # gather text from one hot vector encoding columns
    text = []
    for col in bow_columns:
        if row[col] == 1:
            word = col[5:]
            text.append(word)

    metadata_text = []

    # if the column is in the metadata we provided at the top
    for col in df:
        if col not in bow_columns and col not in non_use_cols \
                and col in metadata:
            # append the name of the column to the answer
            # if we have a name for the column
            col_name_df = cb_df[cb_df.variable == str(col)]
            if len(col_name_df) == 1:
                col_name = col_name_df.iloc[0].question
                try:
                    value = float(row[col])
                    value = str(int(value))
                except ValueError:
                    value = str(row[col])
                    value = re.sub(r'\s+|\[|\]|\'', '', value)

                covariate_name = re.sub(r'\s+|\[|\]|\'', '', col_name)
                metadata_text.append(covariate_name + value)

    datum = {
        "label": row['gs_text34'],
        "text": ' '.join(text),
        "metadata_text": ' '.join(metadata_text),
        "site": row['site'],
        "tariff": tariff[index],
        "interva": interva[index],
        "nbc": nbc[index],
        "insilico": insilico[index]
    }

    # write all the outputs to the same jsonl file
    test_jsonl.write(json.dumps(datum) + "\n")