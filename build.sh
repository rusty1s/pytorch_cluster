#!/bin/sh

echo "Compiling kernel..."

if [ -z "$1" ]; then TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))"); else TORCH="$1"; fi
SRC_DIR=torch_cluster/kernel
BUILD_DIR=torch_cluster/build

mkdir -p $BUILD_DIR
for i in serial grid; do
  $(which nvcc) -c -o "$BUILD_DIR/$i.so" "$SRC_DIR/$i.cu" -arch=sm_35 -Xcompiler -fPIC -shared "-I$TORCH/lib/include/TH" "-I$TORCH/lib/include/THC" "-I$SRC_DIR"
done
