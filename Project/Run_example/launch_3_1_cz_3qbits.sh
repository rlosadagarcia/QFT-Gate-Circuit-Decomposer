#!/bin/bash
#
# this file is myjob.sh
#
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -t 48:00:00 --mem=64G
#SBATCH --job-name 3_1_cz_3q
#SBATCH --mail-type BEGIN,END
#SBATCH --mail-user=roberto.losada@rai.usc.es
#

module load cesga/2020
cd $STORE/Roberto_Losada/Targeted_3_qbits_V2/

python3 -u ./3qbit_cz_3_1.py
