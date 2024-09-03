#!/bin/bash
#SBATCH -J ADG
#SBATCH -p g3090_short
#SBATCH -N 1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH -o %j.out
start_time=$(date +"%Y-%m-%d %H:%M:%S")
echo "Job started at $start_time"
module load compiler/gcc-9.2.0 cuda/11.3
fold=$1
ADG="/home/sangmin/apps/AutoDock-GPU/bin/autodock_gpu_128wi"
JOB="/home/sylee/project/GLP1R/job_file/job${fold}"
echo $JOB
$ADG --filelist $JOB --nrun 50 -x 0