#! /bin/bash
clfile=$1
outfile=$2

punlearn dmextract
command=(pset dmextract infile=$clfile)
"${command[@]}"
pset dmextract outfile=$outfile
pset dmextract opt=generic

{
  expect << EOD
  set timeout 600
  spawn dmextract clobber=yes

  expect {
    "Input event file*" { 
      send "\n" 
      exp_continue 
    }
    "Enter output file name*" { 
      send "\n" 
      exp_continue 
    }
  }
EOD
}
