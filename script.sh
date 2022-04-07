#!/bin/bash


#SBATCH --job-name="CEC2017"


#SBATCH -n 64                # 64 cœurs
#SBATCH -N 2-4               # au minimum 2 nœuds, au maximum 4
#SBATCH --mem=2048           # Quantité mémoire demandée par nœud en Mo (unité obligatoire)
#SBATCH --mail-type=END      # Réception d'un mail à la fin du job
#SBATCH --mail-user=mokhtar.essaid@uha.fr


module load python/python-3.9.7
module rm compilers/intel17

python test.py

