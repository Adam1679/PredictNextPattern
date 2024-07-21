#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi
cd "$1"

# Iterate through all .zip files in the directory
for zipfile in *.zip; do
    unzip -nq "$zipfile"
done
