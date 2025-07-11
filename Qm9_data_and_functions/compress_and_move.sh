#!/bin/bash

# Set your file name
INPUT_FILE="classification_dataset.csv"
OUTPUT_DIR="../data/raw"

# Create the target directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Compress the file using gzip
gzip -c "$INPUT_FILE" > "$OUTPUT_DIR/$INPUT_FILE.gz"

# Optional: remove original file if you don't want to keep it
# rm "$INPUT_FILE"

echo "Compressed and moved to $OUTPUT_DIR/$INPUT_FILE.gz"