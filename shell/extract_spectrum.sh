#! /bin/bash
infile=$1
regfile=$2
stowed=$3
stowedreg=$4
outfile=$5

punlearn specextract
pset specextract infile=$infile'[sky=region('$regfile')]'
pset specextract bkgfile=$stowed'[sky=region('$stowedreg')]'
pset specextract outroot=$outfile
specextract mode=h binarfwmap=4 bkgresp=no clob+


