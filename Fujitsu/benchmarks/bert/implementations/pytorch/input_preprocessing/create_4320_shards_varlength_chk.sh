#! /bin/bash

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
DATADIR=/workspace/bert_data
SHARDS=4320

## Example of how to generate checksums to verify correctness of the process
for i in `seq -w 0000 04319`; do 
  python ${SCRIPT_DIR}/hdf5_md5.py \
    --input_hdf5 ${DATADIR}/hdf5/training-${SHARDS}/hdf5_${SHARDS}_shards_varlength/part_${i}_of_04320.hdf5 
done | tee 4320_shards_varlength_new.chk
