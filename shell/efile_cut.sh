infile=$1
outfile=$2

punlearn dmcopy
dmcopy $infile $outfile clobber=yes
