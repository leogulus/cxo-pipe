#! /bin/bash
arffile=$1
rmffile=$2
modelstr=$3
paramstr=$4
outfile=$5

punlearn modelflux
modelflux arf=$arffile rmf=$rmffile model=$modelstr paramvals=$paramstr emin="0.2" emax="50.0" &>/dev/null
pget modelflux rate pflux flux > $outfile
