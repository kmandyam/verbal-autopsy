#!/usr/bin/env bash

# a script that runs the covariate metadata experiments on the newly modified vampire
python -m scripts.preprocess_data \
            --train-path examples/new_data/train.jsonl \
            --dev-path examples/new_data/dev.jsonl \
            --tokenize \
            --tokenizer-type spacy \
            --vocab-size 30000 \
            --serialization-dir examples/new_data

export DATA_DIR="$(pwd)/examples/new_data"
export VOCAB_SIZE=2492

python -m scripts.train \
            --config training_config/vampire.jsonnet \
            --serialization-dir model_logs/vampire \
            --environment VAMPIRE \
            --device -1 --seed 0

export VAMPIRE_DIR="$(pwd)/model_logs/vampire_0"
export VAMPIRE_DIM=81

mv examples/new_data/train.jsonl examples/new_data/pretrain.jsonl
mv examples/new_data/clftrain.jsonl examples/new_data/train.jsonl

python -m scripts.train \
            --config training_config/classifier.jsonnet \
            --serialization-dir model_logs/clf/weights_ensemble_relu \
            --environment CLASSIFIER \
            --device -1 --seed 0