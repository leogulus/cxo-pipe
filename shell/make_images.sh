#! /bin/bash
infile=$1
outfile=$2
band=$3

fluximage $infile binsize=2 bands=$band outroot=$outfile clobber=yes
