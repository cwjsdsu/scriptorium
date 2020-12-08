#!/usr/bin/python

"""
get_spectrum.py
Jordan Fox , 2016
this program extracts the energy speactrum from an input file (arg 1)
and prints the energies to STDOUT

The input file must be a .res file from BIGSTICK 7.7.0 or similar
Works for BIGSTICK options n, d 

"""

import sys
import os

"""
get input file
"""
if (len(sys.argv)>1):
    yesinfile = True
else:
    yesinfile = False
if (not yesinfile):
    exit("Need input file.")

filename_in = sys.argv[1]    
exists_in = os.path.isfile(filename_in)
if (yesinfile and not exists_in):
    exit(filename_in + " does not exist.")

print(filename_in + " : ")

"""
read
"""
with open(filename_in,"rt") as fptr_in:
    contents = fptr_in.read()

lines = contents.split("\n")

istate = 1
for line in lines:
    try:
    	if len(line.split())==5 and int(line.split()[0])==istate:
        	print(line)
		istate = istate + 1
    except ValueError:
	continue 

