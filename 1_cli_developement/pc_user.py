#!/usr/bin/env python
# coding: utf-8


import os
import pandas as pd
import argparse

def get_slurm_text(full_python_cmd, job_name):
    lizard = job_name
    slurm_text = f'''#!/bin/bash
#SBATCH --partition=amd
#SBATCH --account=butzer
#SBATCH --time=01:59:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=butzer@contractor.usgs.gov
#SBATCH --job-name={lizard}.job
#SBATCH --output=.out/{lizard}.out
#SBATCH --error=.out/{lizard}.err
{full_python_cmd}'''
    return slurm_text

def write_job(slurm_text, city):
    jn = f'.job/{city}.sh'
    f= open(jn,"w+")
    f.write(slurm_text)
    f.close
    return jn

def sbatch_job(job):
    print(job)
    cmd = f'sbatch {job}'
    os.system(cmd)

def make_cmd(h, v, year):
    start = f'{year}0501'
    end = f'{year}0930'
    arguments = f'--horiz {h} --vert {v} --start {start} --end {end}'

    return f'python3 cli_run_composite.py {arguments}'

def do_job(h, v, year):
    cmd = f'python3 ts_user.py --'
    cmd = make_cmd(h, v, year)
    print(cmd)
    job_name=f"{year}_{h}_{v}"
    print(job_name)
    slurm_text = get_slurm_text(cmd, job_name)
    job = write_job(slurm_text, job_name)
    sbatch_job(job)


def _mkdir(directory):
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)

def get_parser():
    parser = argparse.ArgumentParser(description='Run the parallel cluster composite code')
    parser.add_argument('hv_csv_file', metavar='hv_csv_file', type=str, 
            help='the list of H and V up to 527 of them ')
    parser.add_argument('-y', '--year', help='year -y 1999 ', default='2023', type=str, required=True)
    return parser

def run_parallel_cluster(hv_csv_file, year):
    print (hv_csv_file, year)
    df = pd.read_csv(hv_csv_file)

    for i,r in df.iterrows():
        print(r'H')
        print(r'V')
        h = r['H']
        v = r['V']
        do_job(h,v, year)



def command_line_runner():
    parser = get_parser()
    args = vars(parser.parse_args())


    print(args)

    _mkdir('.out')
    _mkdir('.job')

    hv_csv_file = args['hv_csv_file']
    year=args['year']

    run_parallel_cluster(hv_csv_file, year)

    return True


if __name__ == '__main__':

    command_line_runner()
