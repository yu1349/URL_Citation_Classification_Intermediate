#!/bin/bash -x
#PJM -L rscgrp=cx-share
#PJM -L elapse=60:00
#PJM -j
#PJM -S

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ME_241211
python ../src/create_icl.py