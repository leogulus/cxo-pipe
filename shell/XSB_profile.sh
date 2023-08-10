#! /bin/bash
clfile=$1
bkgfile=$2
outfile=$3
outfilermid=$4

punlearn dmextract
command=(pset dmextract infile=$clfile)
"${command[@]}"
command2=(pset dmextract bkg=$bkgfile)
"${command2[@]}"
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

punlearn dmtcalc
pset dmtcalc infile=$outfile
pset dmtcalc outfile=$outfilermid
pset dmtcalc expression="rmid=0.5*(R[0]+R[1])"
{
  expect << EOD
  set timeout 600
  spawn dmtcalc clobber=yes

  expect {
    "Input file*" { 
      send "\n" 
      exp_continue 
    }
    "Output file*" { 
      send "\n" 
      exp_continue 
    }
    "expression*" { 
      send "\n" 
      exp_continue 
    }
  }
EOD
}
