#!/bin/bash

# copy the list of yaml paths from the script into this check:
for f in \
  configs/datasets/mnist/mnist.yml \
  configs/preprocessors/base_preprocessor.yml \
  configs/networks/lenet.yml \
  configs/pipelines/test/test_ood.yml \
  configs/postprocessors/msp.yml \
  configs/datasets/mnist/mnist_ood.yml
do
  echo "== $f =="
  ls -lah "$f" || echo "MISSING!"
  python - <<PY
import yaml, sys
p="$f"
with open(p,"r") as fp:
    obj=yaml.safe_load(fp)
print("loaded type:", type(obj), "is_none:", obj is None)
PY
done
