#! /bin/bash
strrep=$1
dirrep=$2
bands=$3

punlearn merge_obs
merge_obs $strrep $dirrep binsize=2 bands=$bands psfecf=0.95 psfmerge=exptime clobber=yes
