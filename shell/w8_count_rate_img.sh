#! /bin/bash
mapfile=$1
expfile=$2
w8param=$3
outfile=$4

dmimgcalc $mapfile $expfile $outfile div weight=1.0 weight2=$w8param clob+ &>/dev/null
