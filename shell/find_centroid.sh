#! /bin/bash
mapfile=$1
centreg=$2
centfile=$3

dmstat $mapfile'[sky=region('$centreg')]' centroid=yes > $centfile
