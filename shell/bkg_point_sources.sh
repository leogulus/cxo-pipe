#! /bin/bash
dir=$1
regfile=$2

mkdir $dir'/sources'
punlearn roi
pset roi infile=$regfile
pset roi outsrcfile=$dir'/sources/src%d.fits'
pset roi bkgfactor=1.5
pset roi bkgradius=2
pset roi bkgfunction=mul

{
  expect << EOD
  set timeout -1
  spawn roi clob+

  expect {
    "Input src*" { 
      send "\n" 
      exp_continue 
    }
    "Input field*" { 
      send "\n" 
      exp_continue 
    }
    "Input streak*" { 
      send "\n" 
      exp_continue 
    }
    "Output source*" { 
      send "\n" 
      exp_continue 
    }
    "Background radius computation*" { 
      send "\n" 
      exp_continue 
    }
    "Background radius (*" { 
      send "\n" 
      exp_continue 
    }
EOD
}

splitroi $dir'/sources/src*.fits' $dir'/exclude'
dmmakereg 'region('$dir'/exclude.bg.reg)' $dir'/exclude.bg.fits' clob+
