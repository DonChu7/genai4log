#!/bin/bash

# Set dataset name
dataset="linux"

# Create main output directory
output_dir="../output/${dataset}"
if [ -e "$output_dir" ]; then
  echo "$output_dir exists"
else
  mkdir -p "$output_dir"
fi

# Create subdirectory for bert-specific output
bert_output_dir="${output_dir}/bert"
if [ -e "$bert_output_dir" ]; then
  echo "$bert_output_dir exists"
else
  mkdir -p "$bert_output_dir"
fi

# Optional: copy or link raw input log
# Assuming you already placed Linux_2k.log as raw_log.log
# echo "Copying raw log file..."
# cp ../../loghub/Linux/Linux_2k.log ./raw_log.log

