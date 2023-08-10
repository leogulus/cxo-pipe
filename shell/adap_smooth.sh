#! /bin/bash
filein=$1
fileout=$2
filedir=$3

punlearn csmooth
pset csmooth infile=$filein
pset csmooth outfile=$fileout
pset csmooth outsigfile=$filedir'/outsigfile.fits'
pset csmooth outsclfile=$filedir'/outsclfile.fits'
{
  expect << EOD
  set timeout -1
  spawn csmooth sigmin=3 clobber=yes

  expect {
    "input file name*" { 
      send "\n" 
      exp_continue 
    }
    "image of user-supplied map*" { 
      send "\n" 
      exp_continue 
    }
    "output file name*" { 
      send "\n" 
      exp_continue 
    }
    "output significance image*" { 
      send "\n" 
      exp_continue 
    }
    "output scales*" { 
      send "\n" 
      exp_continue 
    }
    "Convolution method*" { 
      send "\n" 
      exp_continue 
    }
    "Convolution kernel*" { 
      send "\n" 
      exp_continue 
    }
    "initial (minimal)*" { 
      send "\n" 
      exp_continue 
    }
    "maximal smoothing*" { 
      send "\n" 
      exp_continue 
    }
    "maximal significance*" { 
      send "\n" 
      exp_continue 
    }
    "compute smoothing*" { 
      send "\n" 
      exp_continue 
    }
  }
EOD
}
