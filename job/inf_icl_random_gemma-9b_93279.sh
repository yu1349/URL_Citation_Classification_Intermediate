#!/bin/bash -x
#PJM -L rscgrp=cxgfs-share
#PJM -L elapse=24:00:00
#PJM -j
#PJM -S

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ME_241211
python ../src/inf_icl.py --icl_method random --seed 93279 --model_name google/gemma-2-9b-it