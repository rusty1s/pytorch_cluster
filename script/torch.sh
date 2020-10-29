#!/bin/bash

# https://github.com/pytorch/pytorch/commit/d2e16dd888a9b5fd55bd475d4fcffb70f388d4f0
if [ "${TRAVIS_OS_NAME}" = "windows" ]; then
  echo "Fix nvcc for PyTorch"
  sed -i.bak -e 's/CONSTEXPR_EXCEPT_WIN_CUDA/const/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/api/module.h
  sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/pybind11/cast.h
fi

# if [ "${TRAVIS_OS_NAME}" = "windows" ] && [ "${TORCH_VERSION}" = "1.7.0" ]; then
#   echo "Fix nvcc for PyTorch 1.7.0"
#   sed -i.bak -e 's/constexpr Symbol Kind = ::c10::prim::profile;/const Symbol Kind;/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/ir/ir.h
#   sed -i.bak '1345a\
# const Symbol ProfileOp::Kind = ::c10::prim::profile;' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/ir/ir.h
#   sed -i.bak -e 's/constexpr Symbol Kind = ::c10::prim::profile_optional;/const Symbol Kind;/g' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/ir/ir.h
#   sed -i.bak '1368a\
# const Symbol ProfileOptionalOp::Kind = ::c10::prim::profile_optional;' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/ir/ir.h

#   echo "-------------------------"
#   sed -n '1327,1375p' /c/tools/miniconda3/envs/test/lib/site-packages/torch/include/torch/csrc/jit/ir/ir.h
#   echo "-------------------------"
# fi
