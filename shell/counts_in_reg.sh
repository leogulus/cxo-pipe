#! /bin/bash
mapfile=$1
regfile=$2
outfile=$3

dmstat $mapfile'[sky=region('$regfile')]' centroid=no > $outfile
