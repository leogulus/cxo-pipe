#! /bin/bash
infile=$1
outfile=$2

mkpsfmap $infile outfile=$outfile energy=1.4967 ecf=0.95 clobber=yes
