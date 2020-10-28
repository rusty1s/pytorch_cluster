#!/bin/bash

# https://github.com/pytorch/pytorch/commit/d2e16dd888a9b5fd55bd475d4fcffb70f388d4f0
if [ "${TRAVIS_OS_NAME}" = "windows" ]; then
  echo "Fix nvcc for PyTorch 1.6.0"
  sed -i.bak -e 's/CONSTEXPR_EXCEPT_WIN_CUDA/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/api/module.h
  sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/pybind11/cast.h
fi
