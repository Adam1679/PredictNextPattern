#!/bin/bash

# Change this to the target directory
# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Change to the target directory where the .zip files are located
target_directory="$1"
postfix="$2"
# Function to remove CSV files
remove_csv_files() {
  local folder="$1"
  echo "Removing CSV files in: $folder"
  find "$folder" -type f -name "*.$postfix" -exec rm -f {} \;
}

# Iterate through all folders in the target directory
for folder in "$target_directory"/*; do
  if [ -d "$folder" ]; then
    remove_csv_files "$folder"
  fi
done
