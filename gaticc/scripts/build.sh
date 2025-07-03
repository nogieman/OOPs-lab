#!/bin/bash

set -e

VENV_DIR=$1
PIP=$VENV_DIR/bin/pip

if [ ! -x "$PIP" ]; then
  echo "[ERROR] pip not found at $PIP"
  exit 1
fi

echo "[BUILD] Pulling latest code..."
git pull origin master

echo "[BUILD] Installing Python requirements..."
$PIP install -r requirements.txt

echo "[BUILD] Installing system dependencies..."
./scripts/install_deps.sh

echo "[BUILD] Cleaning build/ directory..."
rm -rf build

echo "[BUILD] Running CMake configure..."
cmake -DCMAKE_INSTALL_PREFIX=$HOME/.local -B build

echo "[BUILD] Building project..."
cmake --build build -j8

echo "[BUILD] Installing project..."
cmake --install build

echo "[BUILD] Installing Python package (editable)..."
$PIP install -e .

echo "[BUILD] Done!"
