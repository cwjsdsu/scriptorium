#
# script to run compute_M1_pnMEs.py for multiple sampled interaction files
# Fox 7/2021
#

import subprocess as sp
import os
import shutil
import glob
import argparse

script_name ='compute_M1_pnMEs_v2.py'
from shmuq import *

def make_batch_file_content(n_tasks,walltime,error_filename,output_filename,job_name,n_nodes,rs_per_node,tasks_per_rs,cores_per_rs,python_command):
    # NOTE : n_nodes, rs_per_node, tasks_per_node, and cores_per_rs are not used
    # you need to change the file_contents_list if you want those variables included
    file_contents_list=[
    "#!/bin/bash",
    f"#SBATCH -n {n_tasks}           #number of tasks",
    f"#SBATCH -t {walltime}                     #walltime in minutes",
    "#SBATCH -A abinitio                     #account",
    f"#SBATCH -e {error_filename}             #stderr",
    f"#SBATCH -o {output_filename}             #stdout",
    f"#SBATCH -J {job_name}                    #name of job",
    "#SBATCH -p pbatch                   #queue to use",
    "### Shell scripting",
    "date; hostname",
    "echo -n 'JobID is '; echo $LSB_JOBID",
    "conda activate uq",
    "export OMP_NUM_THREADS=8",
    f"mkdir {job_name}",
    f"cp *.x {job_name}",
    f"cp approx.milcom {job_name}",
    f"cp {script_name} {job_name}",
    f"cp shmuq.py {job_name}",
    f"cp sd.sps {job_name}",
    f"cp M1sd_sp.opme {job_name}",
    f"cp M1sd_lp.opme {job_name}",
    f"cp M1sd_sn.opme {job_name}",
    f"cp M1sd_ln.opme {job_name}",
    f"cp sd_M1_processed.csv {job_name}",
    f"cp usdb* {job_name}",
    f"cd {job_name}",
    python_command,
    f"cd ..",
    f"python tar_batch_dir.py {job_name}",
    "echo 'Done'"
    ]
    return "\n".join(file_contents_list)

# for parallel run
#    f"jsrun -N {n_nodes} -a {rs_per_node} -c {tasks_per_rs}  -r {cores_per_rs} -l CPU-CPU -d packed -b packed:1 {python_command}",


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('initial_run_number',type=int)
    parser.add_argument('n_runs_total',type=int)

    parser.add_argument('initial_batch_number',type=int)
    parser.add_argument('n_batches_total',type=int)

    args = parser.parse_args()

    if (args.n_runs_total % args.n_batches_total) != 0:
        exit('n_batches_total does not divide n_runs_total')
    n_runs_per_batch = args.n_runs_total//args.n_batches_total

    input_filename_csv = 'sd_M1_processed.csv'
    batch_counter = 0
    for batch_number in range(args.initial_batch_number,args.initial_batch_number + args.n_batches_total ):
        batch_counter+=1
        n_tasks = 1
        walltime = str(20*n_runs_per_batch)
        batch_name = 'batch_job_' + str(batch_number).zfill(5)
        error_filename = f'myerrors_' + batch_name + '.txt'
        output_filename = f'myoutputs_' + batch_name + '.txt'
        n_nodes = 1
        rs_per_node = 1
        tasks_per_rs = 1
        cores_per_rs = 8  # these are not implemented currently!

        total_python_command_list = []

        current_sample_number_range = range(args.initial_run_number + n_runs_per_batch*(batch_counter-1),args.initial_run_number + n_runs_per_batch*(batch_counter))
        for sample_number in current_sample_number_range:
            sample_number_str = str(sample_number).zfill(index_digits)
            sample_int_name = 'usdb_rand'+sample_number_str
            output_filename_csv = f'sd_M1_{sample_int_name}.csv'
            python_command_list = ['python',script_name,'-i',input_filename_csv,'-o',output_filename_csv,'-s',str(sample_number)]
            #python_command_list = ['python',script_name,input_filename_csv,output_filename_csv,str(sample_number)]
            python_command = " ".join(python_command_list)
            total_python_command_list.append(python_command)
            total_python_command_list.append(f"mkdir run_{sample_int_name}\nmv *rand{sample_number_str}* run_{sample_int_name}")

        total_python_command = "\ndate;\n\n".join(total_python_command_list) + "\ndate;\n\n"
        batch_file_content = make_batch_file_content(n_tasks,walltime,error_filename,output_filename,batch_name,n_nodes,rs_per_node,tasks_per_rs,cores_per_rs,total_python_command)
        batch_filename = batch_name + '.batch'
        with open(batch_filename,'w') as fh:
            fh.write(batch_file_content)
        print('File written:',batch_filename)




