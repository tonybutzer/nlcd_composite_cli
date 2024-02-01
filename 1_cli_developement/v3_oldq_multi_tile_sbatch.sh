#!/bin/sh

job_directory=$PWD/.job

for h in {21..26}; do 
	for v in {11..14} ; do 
		for yr in {1999..2004}; do
		lizard=${h}_${v}_${yr}
    		job_file="${job_directory}/${lizard}.job"
    		echo "#!/bin/bash
#SBATCH --partition=old
#SBATCH --account=butzer
#SBATCH --time=00:20:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=butzer@contractor.usgs.gov
#SBATCH --job-name=${lizard}.job
#SBATCH --output=.out/${lizard}.out
#SBATCH --error=.out/${lizard}.err
python3 cli_run_composite.py --start ${yr}0501 --end ${yr}0930 --horiz ${h} --vert ${v}" > $job_file
    sbatch $job_file

    		done
	done
done
