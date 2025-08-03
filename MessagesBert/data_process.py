import sys
sys.path.append('../')

import os
import re
import json
import pandas as pd
import ast
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
from logparser import Spell, Drain


# === Configuration ===
input_dir = os.path.abspath('./')  # path to LinuxBERT folder
output_dir = os.path.abspath('../output/linux/')
log_file = 'raw_log.log'
log_structured_file = os.path.join(output_dir, log_file + '_structured.csv')
log_templates_file = os.path.join(output_dir, log_file + '_templates.csv')


# === Step 1: Map template hashes to IDs ===
def mapping():
    log_temp = pd.read_csv(log_templates_file)
    log_temp.sort_values(by=["Occurrences"], ascending=False, inplace=True)
    log_temp_dict = {event: idx+1 for idx, event in enumerate(log_temp["EventId"])}
    print(log_temp_dict)
    with open(os.path.join(output_dir, "hdfs_log_templates.json"), "w") as f:
        json.dump(log_temp_dict, f)


# === Step 2: Parse raw logs using Drain ===
def parser(input_dir, output_dir, log_file, log_format, type='drain'):
    if type == 'spell':
        # You can implement Spell here if needed
        pass
    elif type == 'drain':
        regex = [
            r"(?<=blk_)[-\d]+",  # block_id
            r'\d+\.\d+\.\d+\.\d+',  # IP
            r"(/[-\w]+)+",  # file path
        ]
        st = 0.5  # Similarity threshold
        depth = 5
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=False)
        parser.parse(log_file)


# === Step 3: Extract event sequences using sliding window ===
def sliding_window_sampling(log_file, window_size=5, step_size=1):
    print("Sliding window sampling:", log_file)
    df = pd.read_csv(log_file, engine='c', na_filter=False, memory_map=True)
    with open(os.path.join(output_dir, "hdfs_log_templates.json"), "r") as f:
        event_num = json.load(f)
    df["EventId"] = df["EventId"].apply(lambda x: event_num.get(x, -1))

    sequences = []
    for i in range(0, len(df) - window_size + 1, step_size):
        seq = df["EventId"].iloc[i:i+window_size].tolist()
        sequences.append((f"seq_{i}", seq))

    data_df = pd.DataFrame(sequences, columns=["BlockId", "EventSequence"])
    log_sequence_file = os.path.join(output_dir, f"{log_file.split('.')[0]}_sequence.csv")
    data_df.to_csv(log_sequence_file, index=False)
    print("Sliding window sampling done.")


# === Step 4: Split labeled sequences into train, test sets ===
def generate_train_test(hdfs_sequence_file, n=None, ratio=0.3):
    blk_label_dict = {}
    blk_label_file = os.path.join(input_dir, "labeled_logs.csv")
    blk_df = pd.read_csv(blk_label_file)

    # Use line index as BlockId
    for idx, row in tqdm(blk_df.iterrows()):
        blk_label_dict[f"seq_{idx}"] = int(row["label"])

    seq = pd.read_csv(hdfs_sequence_file)
    seq["Label"] = seq["BlockId"].apply(lambda x: blk_label_dict.get(x, -1))  # default -1 if not found

    normal_seq = seq[seq["Label"] == 0]["EventSequence"]
    normal_seq = normal_seq.sample(frac=1, random_state=20)  # shuffle

    abnormal_seq = seq[seq["Label"] == 1]["EventSequence"]
    normal_seq = normal_seq[~normal_seq.isin(abnormal_seq)]

    normal_len, abnormal_len = len(normal_seq), len(abnormal_seq)
    train_len = n if n else int(normal_len * ratio)

    print(f"normal size {normal_len}, abnormal size {abnormal_len}, training size {train_len}")

    train = normal_seq.iloc[:train_len]
    test_normal = normal_seq.iloc[train_len:]
    test_abnormal = abnormal_seq

    df_to_file(train, os.path.join(output_dir, "train"))
    df_to_file(test_normal, os.path.join(output_dir, "test_normal"))
    df_to_file(test_abnormal, os.path.join(output_dir, "test_abnormal"))
    print("generate train test data done")


# === Step 5: Write sequences to flat files ===
def df_to_file(df, file_name):
    with open(file_name, 'w') as f:
        for _, row in df.items():
            try:
                tokens = ast.literal_eval(row)
                if isinstance(tokens, list) and all(isinstance(t, int) for t in tokens):
                    f.write(' '.join(map(str, tokens)) + '\n')
            except Exception as e:
                print(f"Skipping malformed line: {row} ({e})")


# === Run ===
if __name__ == "__main__":
    log_format = '<Month> <Day> <Time> <Host> <Component>\[<Pid>\]: <Content>'
    parser(input_dir, output_dir, log_file, log_format, 'drain')
    mapping()
    sliding_window_sampling(log_structured_file)

    log_sequence_file = os.path.join(output_dir, "raw_log_sequence.csv")
    generate_train_test(log_sequence_file, ratio=0.7)  # or n=10 if you want exact size
