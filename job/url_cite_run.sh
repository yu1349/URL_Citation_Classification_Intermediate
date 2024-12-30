#!/bin/bash -x
#PJM -L rscgrp=cx-share
#PJM elapse=24:00:00
#PJM -j
#PJM -S

eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda activate ME_241211
python ./url_cite_run.py