#!/bin/bash

#PJM -L rscunit=cx
#PJM -L rscgrp=cx-single
#PJM -L node=1
#PJM -L elapse=1:00:00
#PJM -j

module load oneapi

export OMP_NUM_THREADS=40
export OMP_PROC_BIND=spread
# export KMP_AFFINITY=compact,verbose

./stream.exe
