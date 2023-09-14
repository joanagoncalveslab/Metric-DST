#!/bin/sh
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH -o outerr/%j.out # STDOUT
#SBATCH -e outerr/%j.err # STDERR

myfinalcommand="python3 src/emiel_pipeline_wholeset.py"
myfinalcommand+=" $@"

echo $myfinalcommand
srun $myfinalcommand #python3 src/experiments/DBaCT_keras_experiment.py -m $1 -et $2 -th $3 -c $4 -vb -es
