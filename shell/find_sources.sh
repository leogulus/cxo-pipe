#! /bin/bash
mapfile=$1
psffile=$2
expfile=$3
dir=$4

punlearn wavdetect
pset wavdetect infile=$mapfile
pset wavdetect psffile=$psffile
pset wavdetect expfile=$expfile

pset wavdetect outfile=$dir'wide_wdect_src.fits'
pset wavdetect scellfile=$dir'wide_wdect_expmap_scell.fits'
pset wavdetect imagefile=$dir'wide_wdect_expmap_imgfile.fits'
pset wavdetect defnbkgfile=$dir'wide_wdect_expmap_nbgd.fits'
pset wavdetect regfile=$dir'wide_wdect_expmap_src.reg'
{
  expect << EOD
  set timeout -1
  spawn wavdetect clobber=yes

  expect {
    "Input file*" { 
      send "\n" 
      exp_continue 
    }
    "Output source list*" { 
      send "\n" 
      exp_continue 
    }
    "Output source cell*" { 
      send "\n" 
      exp_continue 
    }
    "Output reconstructed*" { 
      send "\n" 
      exp_continue 
    }
    "Output normalized*" { 
      send "\n" 
      exp_continue 
    }
    "wavelet scales*" { 
      send "2. 3. 4. 5.\n" 
      exp_continue 
    }
    "Image of the*" { 
      send "\n" 
      exp_continue 
    }
  }
EOD
}
