#!/bin/sh

echo "Compiling kernel..."

if [ -z "$1" ]; then TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))"); else TORCH="$1"; fi
SRC_DIR=aten/THC
BUILD_DIR=torch_cluster/_ext

mkdir -p $BUILD_DIR
$(which nvcc) -c -o "$BUILD_DIR/THC.so" "$SRC_DIR/THC.cu" -arch=sm_52 -Xcompiler -fPIC -shared "-I$TORCH/lib/include/TH" "-I$TORCH/lib/include" "-I$SRC_DIR"
