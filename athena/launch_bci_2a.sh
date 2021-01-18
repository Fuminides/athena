#!/bin/bash
#$ -q cal.q
#$ -cwd

source activate py365
python ./jupyter_experiments.py $@ >./salidas/salida_2a_$1.txt 2>./salidas/error_2a_$1.txt
