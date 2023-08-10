#! /bin/bash
infile=$1
outfile=$2
gtifile=$3
outclean=$4

punlearn dmextract
pset dmextract infile=$infile'[bin time=::259.28]'
pset dmextract outfile=$outfile
pset dmextract opt=ltc1
{
  expect << EOD
  set timeout -1
  spawn dmextract clobber=yes

  expect {
    "Input event*" { 
      send "\n" 
      exp_continue 
    }
    "Enter output file*" { 
      send "\n" 
      exp_continue 
    }
  }
EOD
}

deflare $outfile $gtifile method=clean

dmcopy $infile"[@"$gtifile"]" $outclean clob+
