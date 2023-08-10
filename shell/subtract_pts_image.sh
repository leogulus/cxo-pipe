#! /bin/bash
infile=$1
outfile=$2
ptsfile=$3
bkgfile=$4

dmfilth $infile $outfile POISSON @$ptsfile @$bkgfile randseed=0 clob+ verbose=0 &>/dev/null
