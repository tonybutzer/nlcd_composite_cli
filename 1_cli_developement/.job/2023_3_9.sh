#!/bin/bash
#SBATCH --partition=amd
#SBATCH --account=butzer
#SBATCH --time=01:59:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=butzer@contractor.usgs.gov
#SBATCH --job-name=2023_3_9.job
#SBATCH --output=.out/2023_3_9.out
#SBATCH --error=.out/2023_3_9.err
python3 cli_run_composite.py --horiz 3 --vert 9 --start 20230501 --end 20230930