#!/bin/bash

# Find and remove files with .out or .err extensions recursively starting from the current directory
find . -type f \( -name "*.out" -o -name "*.err" \) -exec rm -f {} +

echo "Files with .out and .err extensions have been removed."

