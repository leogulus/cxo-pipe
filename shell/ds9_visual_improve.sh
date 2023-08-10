#! /bin/bash

xpaset -p ds9 cmap invert yes
xpaset -p ds9 smooth sigma 3
xpaset -p ds9 smooth radius 5
xpaset -p ds9 smooth
xpaset -p ds9 regions select all
xpaset -p ds9 regions color black
xpaset -p ds9 regions width 2
