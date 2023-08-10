#! /bin/bash
shopt -s expand_aliases
mapfile=$1
regfile=$2

ds9 $mapfile -scale log -cmap b -region $regfile &
