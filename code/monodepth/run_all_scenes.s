#!/bin/bash
for i in $(seq 107 133)
do
sbatch get_depth_data.s $i
done
