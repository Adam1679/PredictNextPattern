#!/bin/bash

# File to parse
FILE_NAME=$1

# Regular expression pattern
PATTERN="wrong value"

# Use grep to filter lines matching the pattern, 
# then use awk to get the values following "wrong value",
# then use sort and uniq to get unique values
values=$(grep -E "$PATTERN" $FILE_NAME | awk -F 'wrong value ' '{print $1}' | uniq | sort | uniq)
echo $values
