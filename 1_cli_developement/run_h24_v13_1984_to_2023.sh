#! /bin/bash

for yr in {1984..2023}; do 
	echo $yr; 
	python3 cli_run_composite.py --start ${yr}0501 --end ${yr}0930 --horiz 24 --vert 13
done

