#!/bin/bash

# Check if a directory argument is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi
target_directory="$1"
# Change to the target directory where the .zip files are located
unzip_files() {
  local folder="$1"
  echo "Unzip files in: $folder"
  find "$folder" -type f -name "*.zip" -exec unzip -nq {} \;
}

# Iterate through all folders in the target directory
for folder in "$target_directory"/*; do
  if [ -d "$folder" ]; then
    unzip_files "$folder"
  fi
done

# # Iterate through all .zip files in the directory
# for zipfile in *.zip; do
#     unzip -nq "$zipfile"
# done
