#
# preprocessing for gamma data
# input ~ sd_E2_data.csv
# output ~ sd_E2_processed.csv
# Fox 2020

import numpy as np
import pandas as pd
from fractions import Fraction
import pickle as pkl

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--input_filename',default='sd_M1_data.csv')
parser.add_argument('--output_filename',default='sd_M1_processed.csv')

args = parser.parse_args()

fn_in = args.input_filename
fn_out = args.output_filename

df = pd.read_csv(fn_in,header=1)

# get rid of empty last column
#df = df.drop(df.columns[[-1]], 1)

# drop any columns with NaNs (should only drop empty columns from conversion of excel format to csv)
df = df.dropna(axis=1, how='all')

df['2Ji'] = df['Ji'].apply(lambda x : int(2*Fraction(x)))
df['2Jf'] = df['Jf'].apply(lambda x : int(2*Fraction(x)))

#df = df.drop('B',axis=1)  # B is Bexp without accounting for intensity, get rid of it

df['Bth'] = 0.  # add column for theory values

#df['Tmirror'] = np.logical_and(df['Zi']==df['Nf'], df['Zf']==df['Ni'])
df['Tmirror'] = (df['Zi']==df['Nf']) & (df['Zf']==df['Ni'])   # note here & does element-wise logic

df['deltaJ'] = 0.5*(df['2Jf']-df['2Ji'])

df = df[(df['2Ji']!=0) | (df['deltaJ']!=0.0)] #remove 100% Fermi transitions

#with open(fn_out,'wb') as fh:
#    pkl.dump(df,fh)


df.to_csv(fn_out,index=False)

