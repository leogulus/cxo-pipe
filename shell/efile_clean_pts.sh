#! /bin/bash
filein=$1
regfile=$2
fileout=$3

dmcopy $filein'[exclude sky=region('$regfile')]' $fileout clobber=yes
