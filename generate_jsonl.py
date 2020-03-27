import os
import re
import json
import pandas as pd
from tqdm import tqdm

# the purpose of this file is to generate jsonl which combines the baseline representations with
# the original representations
# I think the best format is as follows for a given data point

# text: Concatenated text (like with naive metadata)
# label: same as before
# site: same as before
# tariff: the output for tariff
# interva: the output for interva
# nbc: the output for nbc
# insilicova: the output for insilicova

# when we read in the data in our dataset reader, we can more appropriately construct a covariate
# output: test.jsonl, out-domain.jsonl, train.jsonl
# first job: read in the information for each of the baselines

baseline_output_dir = "baseline-representations"
datasets = ['out-domain', 'test', 'train']
baselines = ['tariff2', 'interva2', 'nbc2', 'insilico2']

labels = ["Cancer", "Other NCD", "Diabetes", "Renal", "Stroke", "Liver", "Cardio", "Other Comm",
          "Pneumonia", "TB/AIDS", "Maternal", "External"]

# if it's undetermined, we want to assign equal probability to all labels
def read_baselines(dataset):
    outputs = {}
    for method in baselines:
        results = []
        with open(baseline_output_dir + '/' + dataset + '/' + method + '.txt') as f:
            for line in f.readlines():
                line = line.strip().split("\"")
                label = line[-2]
                assert label in labels or label == "Undetermined"
                results.append(label)
        outputs[method] = results

    return outputs


out_domain_br = read_baselines('out-domain')
test_br = read_baselines('test')
train_br = read_baselines('train')

codebook_file = "codebookfile.csv"
cb_df = pd.read_csv(codebook_file)

metadata_file = "metadata.txt"

metadata = []
with open(metadata_file) as f:
    for line in f:
        metadata.append(line.strip())

def generate_jsonl(csv_path, output_dir, output_name, baseline_rep):
    print("Processing: ", csv_path)
    df = pd.read_csv(csv_path)

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    jsonl_file = open(output_dir + '/' + output_name + ".jsonl", "w", encoding='utf-8')

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
            "tariff": baseline_rep['tariff2'][index],
            "interva": baseline_rep['interva2'][index],
            "nbc": baseline_rep['nbc2'][index],
            "insilico": baseline_rep['insilico2'][index]
        }

        jsonl_file.write(json.dumps(datum) + "\n")
    print()


generate_jsonl('test.csv', 'output-jsonl', 'test', test_br)
generate_jsonl('train.csv', 'output-jsonl', 'train', train_br)
generate_jsonl('out-domain.csv', 'output-jsonl', 'out-domain', out_domain_br)









