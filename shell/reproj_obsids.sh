#! /bin/bash
strrep=$1
dirrep=$2

punlearn reproject_obs
reproject_obs $strrep $dirrep clobber=yes
