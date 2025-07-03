#!/bin/bash

set -e

if [ -z "$1" ]; then
  echo "Usage: $0 <destination directory>"
  exit 1
fi

dest_dir="$1"

# Define the URL of the models.zip file
models_url="http://galactos.local:8471/trained/models.zip"

echo "Downloading models from $models_url..."
wget "$models_url" -O models.zip

echo "Creating directory: $dest_dir/models"
mkdir -p "$dest_dir/models"

echo "Unzipping models.zip into $test_dir/models..."
unzip models.zip -d "$dest_dir/models"

echo "Cleaning up downloaded zip file..."
rm models.zip

echo "Models successfully downloaded and extracted to "$dest_dir"/models" 
