#! /bin/bash
newfile=$1

xpaset -p ds9 regions select all
xpaset -p ds9 regions format ciao
xpaset -p ds9 regions system physical
xpaset -p ds9 regions save $newfile
