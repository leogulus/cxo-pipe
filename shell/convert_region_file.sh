#! /bin/bash
mapfile=$1
regfile=$2
newregfile=$3

punlearn dmmakereg
dmmakereg 'region('$regfile')' $newregfile  wcsfile=$mapfile clob+

