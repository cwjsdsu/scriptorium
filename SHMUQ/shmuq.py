#
#   SHMUQ
#   SHell Model Uncertainty Quantification
#   Fox 2022
#   with options for Cori at NERSC 3/22
#

print("""
  _____ _     __  __ _    _  ____    _
 / ____| |   |  \/  | |  | |/ __ \  | |
| (___ | |__ | \  / | |  | | |  | | | |
 \___ \| '_ \| |\/| | |  | | |  | | |_|
 ____) | | | | |  | | |__| | |__| |  _
|_____/|_| |_|_|  |_|\____/ \___\_\ |_|
Shell Model Uncertainty Quantification
         using BIGSTICK
      < Jordan Fox | 2022 >
""")

from cmath import exp
import numpy as np
import os, sys
import subprocess
import time
import pdb
import pandas as pd
import shutil
import sys
import glob
import operator

verbose = True

core_Z = 8
core_N = 8
max_Z = 20
max_N = 20
core_A = core_Z + core_N

int_name = 'usdb'
int_vec = 'usdb.vec'
sps = "sd"
n_params = 66

op_name = {
        'GT': 'GTsd',
        'beta+^2':'GTsd_beta+_sqr',
        'beta-^2':'GTsd_beta-_sqr',
        'E2(1)':'E2sd_unity',
        'E2(1)^2':'E2sd_unity_sqr',
        'E2_p':'E2sd_p',
        'E2_n':'E2sd_n',
        'E2_p^2':'E2sd_p_sqr',
        'E2_n^2':'E2sd_n_sqr',
        'T^2':'T2sd',
        'M1_sp':'M1sd_sp',
        'M1_sn':'M1sd_sn',
        'M1_lp':'M1sd_lp',
        'M1_ln':'M1sd_ln',
        }

diag_opt = "ld"
n_keep_default = 40

parallel = True
collect_only = False

index_digits = 5 #number of digits in a sample index number
unperturbed_results_dir_E2 = '/p/lustre2/fox39/shmuq/usdb/shmuq_e2/run_usdb'
unperturbed_results_dir_GT = '/p/lustre2/fox39/shmuq/usdb/shmuq_gt/run_usdb'
unperturbed_results_dir_M1 = '/p/lustre2/fox39/shmuq/usdb/shmuq_m1/run_usdb'
milcoms_filename = 'approx.milcom'
int_mil_vec = 'usdbmil.vec'

bigstick_serial = './bigstick.x'
bigstick_omp = './bigstick-openmp.x'
bigstick_mpi_omp = './bigstick-mpi-omp.x'
bigstick_knl = './bigstick-mpi-omp_knl.x'

anamil_cmd = "./anamil_v4.x"
standard_cmd = "./standardorderint.x"
gtstrength_cmd = "./gtstrength.x"
genstrength_cmd = "./genstrength_mod.x"
use_openmp = True
use_mpi = False

if use_openmp:
    include_frag = False
    bigstick_cmd = bigstick_omp
if use_mpi:
    include_frag = True
    #bigstick_cmd = bigstick_mpi_omp
    bigstick_cmd = bigstick_knl
if not use_openmp and not use_openmp:
    include_frag = False
    bigstick_cmd = bigstick_serial

##############################################################
# print SHMUQ config

if verbose:
    print('Your current options are as follows')
    print(f'Core N,Z = {core_N} , {core_Z}')
    print(f'Maximum N,Z = {max_N} , {max_Z}')
    print(f'Single particle space: {sps}')
    print(f'Interaction (Hamiltonian): {int_name}')
    print(f'Number of Hamiltonian components: {n_params}')
    print('Please check module script for more options!')
    print("ShMUQ does not currently set any environment variables.\nIf you want OMP_NUM_THREADS, you need to set it yourself.")

###################################################################

# element name dict as fxn of Z
element_name = {
        1:'H',
        2:'He',
        3:'Li',
        4:'Be',
        5:'B',
        6:'C',
        7:'N',
        8:'O',
        9:'F',
        10:'Ne',
        11:'Na',
        12:'Mg',
        13:'Al',
        14:'Si',
        15:'P',
        16:'S',
        17:'Cl',
        18:'Ar',
        19:'K',
        20:'Ca',
        21:'Sc',
        22:'Ti',
        23:'V',
        24:'Cr',
        25:'Mn',
        26:'Fe',
        27:'Co',
        28:'Ni',
        29:'Cu',
        30:'Zn',
        31:'Ga',
        32:'Ge',
        33:'As',
        34:'Se',
        35:'Br',
        36:'Kr',
        37:'Rb',
        38:'Sr',
        39:'Y',
        40:'Zr',
        41:'Nb',
        42:'Mo',
        43:'Tc',
        44:'Ru',
        45:'Rh',
        46:'Pd',
        47:'Ag',
        48:'Cd',
        49:'In',
        50:'Sn',
        }

class nucleus:
    """ Holds information about the nucleus and BIGSTICK calculation."""
    def __init__(self,Z,N,twoJz):
        self.Z = int(Z)
        self.N = int(N)
        self.Zv = self.Z - core_Z
        self.Nv = self.N - core_N
        self.A = self.Z + self.N
        self.element = element_name[self.Z]
        self.name = str(self.A) + self.element
        self.scaling = ['1',str(core_A+2),str(self.A),'0.3']   # note this should probably be changed for different Hamiltonians
        self.twoJz = int(twoJz)
        self.twoTz = self.Z - self.N

    def __str__(self):
        s_list = [f'NUCLEUS:  {self.name}',
                f'Z = {self.Z},  N = {self.N}',
                f'Zv = {self.Zv},  Nv = {self.Nv}',
                f'2*Jz = {self.twoJz},  2*Tz = {self.twoTz}',
                f'Interaction scaling = ' + ' '.join(self.scaling),
                ]
        return "\n".join(s_list)+"\n"

# functions get_parent and get_daughter are meant to get info from a pandas DF of 1 or more transitions

def get_parent(t):
    """ Takes in a Pandas DataFrame row and returns the parent Nucleus object.
    Relevant for charge-changing transitions.
    Input must have attributes 'Zi' and 'Ni'.
    """
    #Z = t['Zi'].iloc[0]
    #N = t['Ni'].iloc[0]
    Z = t['Zi']
    N = t['Ni']
    twoJz = t['twoJi'].min()
    return nucleus(Z,N,twoJz)

def get_daughter(t):
    """ Takes in a Pandas DataFrame row and returns the daughter Nucleus object.
    Relevant for charge-changing transitions.
    Input must have attributes 'Zf' and 'Nf'.
    """
    #Z = t['Zf'].iloc[0]
    #N = t['Nf'].iloc[0]
    Z = t['Zf']
    N = t['Nf']
    twoJz = t['twoJf'].min()
    return nucleus(Z,N,twoJz)

def how_many_lanczos_iterations(n_keep):
    """ Rule to get # of Lanczos interations from n_keep """
    return max(200,40*n_keep)

def get_nucleus(t):
    """ Takes in a Pandas DataFrame and returns the parent Nucleus object.
    Use this for non charge-changing transitions.
    """
    # automatically take minimum 2Jz
    return nucleus(t['Z'],t['N'],int(t['A'])%2)

#
#   LITTLE TOOLS
#

def remove_suffix(s,suffix):
    """ Does what it says on the tin. """
    if s.endswith(suffix):
        idx = -1*len(suffix)
        return s[:idx]
    else:
        return s

def times_so_far(ls):
    """ Takes a list and returns a list of the same size with the number of times that element has appeared in the list so far.
    Used for computing n for excited states : J^pi_n
    """
    out = [0]*len(ls)
    for i in range(len(ls)):
        out[i] = ls[:i+1].count(ls[i])
    return out

def round_to_half(x):
    return round(2*x)/2

def getspectrum(filename):
    """ Reads bigstick file and prints the spectrum"""
    
    if verbose: print('Getting spectrum from '+filename)
    outs = []
    if filename.endswith('.res'):
        pass
    else:
        filename+='.res'
    with open(filename,"rt") as fptr_in:
        contents = fptr_in.read()
        lines = contents.split("\n")
        istate = 1
        for line in lines:
            try:
                if (len(line.split())==5 or len(line.split())==6) and int(line.split()[0])==istate:
                    outs.append(line.split())
                    istate = istate + 1
            except ValueError:
                continue
    if verbose: print('Got spectrum from '+filename)
    return outs

# alias
def get_spectrum(filename):
    """ Reads bigstick file and prints the spectrum"""
    return getspectrum(filename)

def count_duplicates(spec,match_columns,round_columns=None): #adds last column to specrum counting duplicate states
    """ Takes an array 'spec' and adds a column to the end that is n (as in J^pi_n)
    """
    if round_columns is not None:
        for col_num in round_columns:
            spec[:,col_num] = [str(round_to_half(float(x))) for x in spec[:,col_num]]
    counts = times_so_far(list([list(tj) for tj in spec[:,match_columns]]))
    counts = np.array(counts)
    counts = counts.reshape(-1,1)
    return np.append(spec,counts,axis=1)

def cleanup():
    """ Deletes some files that are not useful.
    Probably don't run this function...
    """
    cleanup_cmd = " ".join(["rm","*.lcoef","timingdata.bigstick",\
        "timinginfo.bigstick","fort.*"])
    subprocess.Popen(cleanup_cmd,shell=True)

def exists(fn, blow_up=False):
    x = os.path.exists(fn)
    if x:
        print(f'Found file: {fn}')
    else:
        print(f'WARNING! File does not exist: {fn}')
        if blow_up:
            exit('Blowing up now.')
    return x


def remove_file(fn):
    if exists(fn):
        print(f'Removing file: {fn}')
        os.remove(fn)


#
# BIGSTICK stuff
#

def make_bigstick_input(input_int,nuc,opt,twoJz=None,n_keep=n_keep_default,Tshift=0.,operator=None):
    """
    opt = n, d, or x
    for option v use other fxn: compute_overlaps

    input_int is the input to this function. int_name is global interaction for UQ
    i.e. int_name=gx1a , input_int = gx1a_with_perturbation

    nuc is the Nucleus object

    operator is only for option x
    DO NOT CONFUSE THIS WITH THE GLOBAL VARIABLE op_name, which is a dict

    """
    print('Writing bigstick inputs...')

    if twoJz is None:
        twoJz = nuc.twoJz

    #exact_diag_condition = ((nuc.Zv<2) & (nuc.Nv<2)) | (((max_Z-nuc.Z)<2) & ((max_N-nuc.N)<2))

    dim = model_bigstick_calculation(nuc)
    if dim<500:
        exact_diag_condition = True
    else:
        exact_diag_condition = False
    
    def option_n(input_int,nuc,n_keep,Tshift):
        # option n/d inputs:
        # 1 n
        # 2 output name
        # 3 sps
        # 4 Zv Nv
        # 5 2Jz
        # 6 interaction
        # 7 scaling
        # 8 end
        # 9 lanczos option
        # 10 nkeep, lanczos iterations

        ZvNv = " ".join([str(nuc.Zv),str(nuc.Nv)])
        frag = "0"

        # option for near shell boundaries -> small  dimension
        if exact_diag_condition:
            diag_opt = 'ex'
        else:
            diag_opt = "ld"

        run_name_list = [nuc.name,input_int]
        if Tshift>0.:
            run_name_list.append(f'Tsh{Tshift}')
        run_name_list.append(f'2M{twoJz}')
        run_name = '_'.join(run_name_list)

        scaling = nuc.scaling
        in_list = ['n',run_name,sps,ZvNv,str(twoJz)]
        if include_frag:
            in_list.append(frag)
        in_list = in_list + [input_int," ".join(scaling)]
        if Tshift>0.:
            tshift_scaling = [-1*Tshift,-1*Tshift,0,0]
            in_list = in_list + [op_name['T^2'],' '.join([str(x) for x in tshift_scaling])]

        lanczos_iter = how_many_lanczos_iterations(n_keep)
        lanczos_opt = f'{n_keep} {lanczos_iter}'
        in_list = in_list + ["end",diag_opt,lanczos_opt]
        if verbose: print(f'n_keep = {n_keep}')

        return in_list, run_name

    def option_d(input_int,nuc,n_keep,Tshift):
        # option n/d inputs:
        # 1 n
        # 2 output name
        # 3 sps
        # 4 Zv Nv
        # 5 2Jz
        # 6 interaction
        # 7 scaling
        # 8 end
        # 9 lanczos option
        # 10 nkeep, lanczos iterations

        ZvNv = " ".join([str(nuc.Zv),str(nuc.Nv)])
        frag = "0"

        # option for near shell boundaries -> small  dimension
        if exact_diag_condition:
            diag_opt = 'ex'
        else:
            diag_opt = "ld"

        run_name_list = [nuc.name,input_int]
        if Tshift>0.:
            run_name_list.append(f'Tsh{Tshift}')
        run_name_list.append(f'2M{twoJz}')
        run_name = '_'.join(run_name_list)

        scaling = nuc.scaling
        in_list = ['d',run_name,sps,ZvNv,str(twoJz)]
        if include_frag:
            in_list.append(frag)
        in_list = in_list + [input_int," ".join(scaling)]
        if Tshift>0.:
            tshift_scaling = [-1*Tshift, -1*Tshift, 0, 0]
            in_list = in_list + [ op_name['T^2'] , ' '.join([str(x) for x in tshift_scaling]) ]

        lanczos_iter = how_many_lanczos_iterations(n_keep)
        lanczos_opt = f'{n_keep} {lanczos_iter}'
        in_list = in_list + ["end",diag_opt,lanczos_opt]
        if verbose: print(f'n_keep = {n_keep}')

        return in_list, run_name

    def option_x(input_int,nuc,operator):
        # option x inputs (xpn format)
        # 1 x
        # 2 input (wfn) name
        # 3 output name
        # 4 xpn
        # 5 operator name
        # 6 scaling
        # 7 xpn scaling
        # 8 end

        if operator==None:
            exit(f'ERROR: for this option you must enter an operator <operator>. Terminating.')
        elif not exists(f'{operator}.int'):
            exit('ERROR: operator file does not exist: {operator}.int ... Terminating.')

        input_name = '_'.join([nuc.name,input_int,f'2M{twoJz}'])
        run_name = input_name + f'_X_{operator}'

        scaling = ' '.join(['1']*4)
        xpn_scaling = ' '.join(['1']*5)

        xpn_format = True # operator in xpn format

        in_list = ['x',input_name,run_name]
        if xpn_format:
            in_list = in_list + ['xpn',operator,scaling,xpn_scaling,'end']
        else:
            in_list = in_list + [operator,scaling,'end']

        return in_list, run_name


    n_keep = min(dim,n_keep)
    if opt=='n':
        in_list, run_name = option_n(input_int,nuc,n_keep,Tshift)
    elif opt=='d':
        in_list, run_name = option_d(input_int,nuc,n_keep,Tshift)
    elif opt=='x':
        in_list, run_name = option_x(input_int,nuc,operator)

    outfn = "in."+run_name+".b"
    with open(outfn,'w') as outfh:
         outfh.write("\n".join(in_list)+"\n")
    if verbose: print(f'Bigstick input written to file : {outfn}')

    return run_name

def make_cori_batch(run_name,nuc):
    """ Makes batch file for Cori (NERSC).
    """

    def dimension(nuc):
        dist = np.sqrt(nuc.Zv**2 + nuc.Nv**2) 
        """ Rough fit to pf-shell dimensions. """
        return np.exp(24 - 30*np.exp(-0.19*dist -0.1))

    n_nodes = 1
    minutes = 10
    dim_guess = dimension(nuc)
    if dim_guess > 1E4:
        n_nodes = 1
        minutes = 10
    if dim_guess > 1E5:
        n_nodes = 1
        minutes = 30
    if dim_guess > 1E6:
        n_nodes = 16
        minutes = 60
    if dim_guess > 1E7:
        n_nodes = 16
        minutes = 100
    if dim_guess > 1E8:
        n_nodes = 64
        minutes = 300
    if dim_guess > 1E9:
        n_nodes = 256
        minutes = 600
    if dim_guess > 1E10:
        n_nodes = 512
        minutes = 600

    arch = 'knl'
    cores = 272
    threads = 8
    bigstick = bigstick_knl
    run_command = f'srun {bigstick} < in.{run_name}.b'
    contents = f"""#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time={minutes}
#SBATCH --nodes={n_nodes}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={cores}
#SBATCH --constraint={arch}
#SBATCH --output={run_name}.batchout
#SBATCH --error={run_name}.batcherr

export OMP_PROC_BIND=true
export OMP_PLACES=cores
export OMP_NUM_THREADS={threads}

{run_command}

"""
    batch_fn = run_name+'.batch'
    with open(batch_fn,'w') as batch_fh:
        batch_fh.write(contents)
    return 

def submit_batch_job(run_name):
    batch_fn = run_name+'.batch'
    print(f'Submitting batch job: {run_name}')
    cmd = f'sbatch {batch_fn}'
    subprocess.call(cmd,shell=True)

def run_bigstick(run_name):
    print(f'Running bigstick for {run_name}')
    cmd = " ".join([bigstick_cmd,"<",f'in.{run_name}.b'])
    subprocess.call(cmd,shell=True)

def model_bigstick_calculation(nuc):
    bargs = ['m','none',sps,f'{nuc.Zv} {nuc.Nv}',str(nuc.twoJz),'0','1','1000']
    brun = subprocess.Popen(bigstick_serial,stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT) 
    brun.stdin.write("\n".join(bargs).encode())
    bout = brun.communicate()[0]
    bout = bout.decode('UTF-8')
    brun.stdin.close()
    if verbose: print('OUTPUT OF MODELING RUN:' + "\n" + bout)
    for line in bout.split("\n"):
        ls = line.split() 
        if ls[:3]==['Total','basis','=']:
            dim = int(ls[-1])
            break
    return dim

def compute_overlaps(wfn_initial,wfn_final,i_state):
    """NOTE: this input to bigstick requires a modification to
     bdenslib1.f90 to prompt for a custom output name
     if you want to not modify Bigstick, use the commented out fxn below
     """

    i_name = wfn_initial.split('/')[-1]
    f_name = wfn_final.split('/')[-1]

    fn_overlaps = f'{i_name}_st{i_state}_{f_name}.ovlp'

    bargs = ['v',wfn_initial,fn_overlaps,str(i_state),wfn_final]
    brun = subprocess.Popen(bigstick_serial,stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    brun.stdin.write("\n".join(bargs).encode())
    bout = brun.communicate()[0]
    bout = bout.decode('UTF-8')
    brun.stdin.close()
    if verbose:
        print('Computing overlaps:',wfn_initial,wfn_final,i_state)
        print('ovlp args:',bargs)
        print(bout)
    with open(fn_overlaps,'r') as fh:
        overlaps = np.loadtxt(fh,skiprows=2)
    return overlaps

#def compute_overlaps(wfn_initial,wfn_final,i_state):
#    bargs = ['v',wfn_initial,str(i_state),wfn_final]
#    bigstick_cmd = bigstick_serial
#    brun = subprocess.Popen(bigstick_cmd,stdin=subprocess.PIPE,
#        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
#    brun.stdin.write("\n".join(bargs).encode())
#    bout = brun.communicate()[0]
#    #brun.wait()
#    bout = bout.decode('UTF-8')
#    brun.stdin.close()
#
#    # rename overlap.dat
#    fn_overlaps = f'o_{wfn_initial}_st{i_state}_{wfn_final}.dat'
#    os.rename('./overlap.dat',fn_overlaps)
#    # read overlaps
#    with open(fn_overlaps,'r') as fh:
#        overlaps = np.loadtxt(fh,skiprows=2)
#    return overlaps

def overlap_index(wfn_initial,wfn_final,i_state):
    """ gets index of state by computing overlaps <initial|final>
     counting starts at zero!
    """
    overlaps = compute_overlaps(wfn_initial,wfn_final,i_state)
    idx = np.argmax(overlaps[:,-1])    # get overlap magnitude |<f|i>|^2
    cutoff = 0.25
    if overlaps[idx,-1]<cutoff:
        exit(f'Detected overlap<{cutoff}: {wfn_initial}, {wfn_final}, i={i_state}. Terminating.')
    return idx

def get_state_index(res_file,J,n):
    """ gets index of a particular state by J,n
    """
    spectrum = np.array(getspectrum(res_file),dtype=float)
    if verbose: print('SPECTRUM:',res_file,spectrum[spectrum[:,3]==J])
    return int(spectrum[spectrum[:,3]==J][n-1,0])

def make_gtstrength_inputs(parent_name,daughter_name,density_name_list,parent,daughter,Tshift=0.):
    """ makes input files for gtstrength

    density_name_list is a list of density file names, max 2

    """
    print('Writing gtstrength inputs...')

    scaling = 1.0

    par_ZN = ' '.join([str(x) for x in [parent.Zv,parent.Nv]])
    dau_ZN = ' '.join([str(x) for x in [daughter.Zv,daughter.Nv]])

    gt_run_name = '_'.join([parent_name,op_name['GT']])

    outfn = "in." + gt_run_name +".gtstr"
    in_list = [op_name['GT'],str(scaling),parent_name,"0",daughter_name,"0"]
    if len(density_name_list)==1:
        in_list +=  [density_name_list[0],f'{Tshift}','n']
    elif len(density_name_list)==2:
        in_list +=  [density_name_list[0],f'{Tshift}','y',density_name_list[1],f'{Tshift}','n']
    in_list += [gt_run_name,par_ZN,dau_ZN]
    with open(outfn,'w') as outfh:
        outfh.write("\n".join(in_list)+"\n")
    return gt_run_name

def make_genstrength_inputs(run_name,int_name,nuc,opme_name):
    """ makes input files for genstrength
    """

    print('Writing genstrength inputs...')

    scaling = 1.0

    ZvNv = " ".join([str(nuc.Zv),str(nuc.Nv)])

    gs_run_name = '_'.join([nuc.name,int_name,f'2M{nuc.twoJ}',opme_name])

    outfn = "in." + gs_run_name +".genstr"
    in_list = [opme_name,str(scaling),run_name,"0",run_name,"0",run_name,'n',run_name+'_'+opme_name]
    with open(outfn,'w') as outfh:
        outfh.write("\n".join(in_list)+"\n")
    return gs_run_name

def make_genstrength_inputs_v2(run_name:str,int_name:str,nuc,opme_name:str,density_list:list):
    """ makes input files for genstrength, but accepts any number of density matrices.
    density_list must be a list, but it can have 1 element"""

    print('Writing genstrength inputs...')

    scaling = 1.0

    ZvNv = " ".join([str(nuc.Zv),str(nuc.Nv)])

    gs_run_name = '_'.join([nuc.name,int_name,opme_name])

    outfn = "in." + gs_run_name +".genstr"
    in_list = [opme_name,str(scaling),run_name,"0",run_name,"0",density_list[0]]
    if len(density_list)>1:
        for dens in density_list[1:]:
            in_list += ['y',dens]
    in_list += ['n',gs_run_name]
    with open(outfn,'w') as outfh:
        outfh.write("\n".join(in_list)+"\n")
    return gs_run_name

def run_genstrength(gs_run_name):
    input_fn = 'in.'+gs_run_name+'.genstr'
    print('Running genstrength_mod for '+input_fn)
    cmd = " ".join([genstrength_cmd,"<",input_fn])
    subprocess.call(cmd,shell=True)

def run_gtstrength(gt_run_name):
    input_fn = 'in.'+gt_run_name+'.gtstr'
    print('Running gtstrength for '+input_fn)
    cmd = " ".join([gtstrength_cmd,"<",input_fn])
    subprocess.call(cmd,shell=True)


def parse_strength_file(gs_run_name,trans_dataframe,key_name, reading_m=False,skip_lines=0):
    """ 
    Updates trans_dataframe with GT information
    This relies on J,n for state index

    key_name should be something like Bth or Mth probably

    reading_m means we read M-values from .mstr files, rather thna B-values from .str files (default)
    """
    if reading_m:
        fn = gs_run_name+'.mstr'  #filename convention
    elif not reading_m:
        fn = gs_run_name+'.str'  #filename convention

    if verbose: print("Parsing "+fn+"...")

    for i_t,t in trans_dataframe.iterrows():
        if verbose: print(f'i_t = {i_t}')
        line_num = 0
        with open(fn,'r') as fh:
            print("Looking for this transition:\n",t)
            counteri = t['ni'] - 1
            counterf = t['nf'] - 1
            found_parent = False
            for line in fh:
                line_num += 1
                if line_num <= skip_lines:
                    continue
                ls = line.split()
                if ('parent' in ls) and (float(ls[2])*2 == t['twoJi']):
                    if (counteri==0):
                        found_parent = True
                        print("found parent :",line)
                    elif (counteri>0):
                        counteri = counteri - 1
                if found_parent and ('daughter' in ls) and (float(ls[2])*2 == t['twoJf']):
                    if (counterf==0):
                        trans_dataframe.loc[i_t,key_name] = float(ls[4])
                        print("found daughter :",line)
                        break
                    elif (counterf>0):
                        counterf = counterf - 1
    return trans_dataframe


def parse_strength_file_by_idx(run_name,state_index_list,trans_dataframe,key_name, reading_m=False, skip_lines=0):
    """ Same as parse_strength_file, eaxcept 
    this reads values out of a str results file based on a pre-existing index list (e.g. from overlaps)

    key_name should be something like Bth or Mth probably
    
    reading_m means we read M-values from .mstr files, rather thna B-values from .str files (default)
    """

    if reading_m:
        fn = run_name+'.mstr'  #filename convention
    elif not reading_m:
        fn = run_name+'.str'  #filename convention

    if verbose: print("Parsing "+fn+"...")

    if verbose: print('State index list:',state_index_list)
    count=0
    for i_t,t in trans_dataframe.iterrows():
        idx_i = state_index_list[count][0]
        idx_f = state_index_list[count][1]
        if verbose: print('count = ',count)
        count+=1

        if idx_i is None or idx_f is None:
            print('-'*50)
            print('ALERT: This transition has been excluded:', t)
            print('-'*50)
            continue


        with open(fn,'r') as fh:
            if verbose: print('looking for this transition:',t)
            if verbose: print(f'i_t = {i_t}')
            line_num = 0
            found_i = False
            found_f = False
            for line in fh:
                line_num += 1
                if line_num <= skip_lines:
                    continue
                ls = line.split()
                #if verbose: print(f'line {line_num}:',line)
                if found_i and ('parent' in ls):
                    # if next parent state is reached w/o finding daughter
                    if verbose: print(f'ERROR: initial state has no final state: {run_name}')
                    sys.exit(f'ERROR: initial state has no final state: {run_name}')
                if ('parent' in ls) and (int(ls[0])==int(idx_i+1)):
                    # found parent
                    found_i = True
                    if verbose: print("found initial state :",line)
                if found_i and ('daughter' in ls) and (int(ls[0])==int(idx_f+1)):
                    found_f = True
                    trans_dataframe.loc[i_t,key_name] = float(ls[4])
                    if verbose: print("found final state :",line)
                    break
            if not found_i:
                print(t)
                sys.exit(f'ERROR: initial state is missing: {run_name}')
            if not found_f:
                print(t)
                sys.exit(f'ERROR: final state is missing: {run_name}')
    return trans_dataframe


def state_indices_pairs_from_ovlp(dataframe,sample_int_name,run_name,results_dir_name):
    """
    get pairs of state indices (within one nucleus) based on overlaps
     requires a directory of prior results to determine proper ordering
     e.g. put all wfns from original into a folder
     state info in dataframe
     sample_int_name is something like usdb_rand_00001
     run_name is the run with the sample int, e.g. 20Ne_usdb_rand_00001_2M0
    """
    index_pairs_list = []
    run_name_unperturbed = os.path.join(results_dir_name,run_name.replace(sample_int_name,int_name))
    if verbose: print(f'RUN NAME UNPERTURBED = {run_name_unperturbed}')
    for i_entry,entry in dataframe.iterrows():
        if verbose: print(f'state_indices_pairs_from_ovlp entry {i_entry}:',entry)
        n_i = int(dataframe.loc[i_entry,'ni'])
        n_f = int(dataframe.loc[i_entry,'nf'])
        j_i = int(dataframe.loc[i_entry,'twoJi'])/2
        j_f = int(dataframe.loc[i_entry,'twoJf'])/2
        try:
            idx_i = get_state_index(run_name_unperturbed,j_i,n_i)
            idx_f = get_state_index(run_name_unperturbed,j_f,n_f)
        except:
            index_pairs_list.append([None,None])
            if verbose:
                print('-'*50)
                print('ALERT: excluding this transition because state indices could not be found:', entry)
                print('-'*50)
            continue
        idx_i = overlap_index(run_name_unperturbed,run_name,idx_i)
        idx_f = overlap_index(run_name_unperturbed,run_name,idx_f)
        index_pairs_list.append([idx_i,idx_f])
    return index_pairs_list

def state_indices_from_ovlp(dataframe,sample_int_name,run_name,results_dir_name):
    """
    get single (initial) state indices (within one nucleus) based on overlaps
     requires a directory of prior results to determine proper ordering
     e.g. put all wfns from original into a folder
     state info in dataframe
     sample_int_name is something like usdb_rand_00001
     """
    index_list = []
    run_name_unperturbed = os.path.join(results_dir_name,run_name.replace(sample_int_name,int_name))
    if verbose: print('state_indices_from_ovlp reference file:',run_name_unperturbed)
    for i_entry,entry in dataframe.iterrows():
        if verbose: print(f'state_indices_from_ovlp entry {i_entry}:',entry)
        n = int(dataframe.loc[i_entry,'ni'])
        j = int(dataframe.loc[i_entry,'twoJi'])/2
        idx_state = get_state_index(run_name_unperturbed,j,n)
        idx = overlap_index(run_name_unperturbed,run_name,idx_state)
        index_list.append(idx)
    return index_list

def parent_daughter_indices(transitions,sample_int_name,run_name_par,run_name_dau,results_dir_name):
    """ state indices for a charge-changing transition (between 2 nuclei)
    """
    index_pairs = []
    parent_name_unperturbed = os.path.join(results_dir_name,run_name_par.replace(sample_int_name,int_name))
    daughter_name_unperturbed = os.path.join(results_dir_name,run_name_dau.replace(sample_int_name,int_name))
    for i_t,t in transitions.iterrows():
        if verbose: print(f'i_t = {i_t}')
        n_parent = int(transitions.loc[i_t,'ni'])
        n_daughter = int(transitions.loc[i_t,'nf'])
        j_parent = int(transitions.loc[i_t,'twoJi'])/2
        j_daughter = int(transitions.loc[i_t,'twoJf'])/2
        i_parent_unperturbed = get_state_index(parent_name_unperturbed,j_parent,n_parent)
        i_daughter_unperturbed = get_state_index(daughter_name_unperturbed,j_daughter,n_daughter)
        i_parent = overlap_index(parent_name_unperturbed,run_name_par,i_parent_unperturbed)
        i_daughter = overlap_index(daughter_name_unperturbed,run_name_dau,i_daughter_unperturbed)
        index_pairs.append([i_parent,i_daughter])
    return index_pairs

def make_sample_interaction(int_name,sample_number,milcoms_filename):
    sample_number = str(sample_number)
    hessian_eigenvalues = np.loadtxt(milcoms_filename,skiprows=1,delimiter="\t")[:n_params]
    param_variance = 1/hessian_eigenvalues
    sample_int_name_list = [int_name,'rand' + sample_number.zfill(index_digits)]
    sample_int_name = '_'.join(sample_int_name_list)
    pert = np.random.multivariate_normal(mean=np.zeros(n_params),cov=np.diag(param_variance))
    pert = np.stack((np.arange(1,n_params+1),pert),axis=-1)
    perturb_milcom(sample_int_name,pert,int_mil_vec)
    return sample_int_name

def move_files_to_folder(sample_int_name,sample_number):
    folder_name = 'run_' + sample_int_name
    codes = {}
    if os.path.isdir(folder_name):
        print(f'Directory {folder_name} already exists.')
    else:
        code_mkdir = subprocess.call(['mkdir',folder_name])
    
    file_list = glob.glob(f'*rand' + str(sample_number).zfill(index_digits) + '*')
    for file_name in file_list:
        shutil.move(file_name,folder_name)
    if code_mkdir==0:
        print(f'Files moved to {folder_name}')
    else:
        sys.exit(f'Error: nonzero exit code in mkdir: {sample_number}')

def read_exp_values(filename):
    if verbose: print('Getting expectation values from '+filename)
    outs = []
    if filename.endswith('.res'):
        pass
    else:
        filename+='.res'
    with open(filename,"rt") as fptr_in:
        contents = fptr_in.read()
    lines = contents.split("\n")
    istate = 1
    for line in lines:
        try:
            if len(line.split())==6 and int(line.split()[0])==istate:
                outs.append(line.split())
                istate = istate + 1
        except ValueError:
            continue
    outs = np.array(outs)
    if verbose: print('Got spectrum from '+filename)
    # NOTE this outputs the whole line: i E J T <H> norm
    col_names = ['Index','Energy','J','T^2','Exp Val','Norm']
    temp = {}
    for k in range(6):
        temp[col_names[k]] = outs[:,k]
    df_out = pd.DataFrame.from_dict(temp)
    return df_out


# Sensitivity Analysis functions


#
#   DATA
#

def get_data(data_fn,max_unc=2.0,abs_energies=True,swap_jt=True,use_pandas=True):
    # data columns = A(0) , Z(1) , N(2) , 2t(3) , 2j(4) , n(5) , energy(6) , error(7)
    data = np.genfromtxt(data_fn, delimiter=',',skip_header=1)
    if max_unc is not None:   # delete data with dE >= max_unc in MeV
        data = data[data[:,7]<max_unc]

    if not abs_energies:
        # if abs_energies is TRUE then all energies are absolute
        # else, assume that g.s. energy is absolute and the rest are excitations
        for i,line in enumerate(data):  #absolute energies
            if line[6]<0:     # if find g.s.
                E0 = line[6]
            elif line[6]>0:    # add g.s. energy to excitations to get absolute
                data[i,6] = line[6] + E0

    if swap_jt:   # if you need to swap j and t columns
        data[:,[3,4]] = data[:,[4,3]]   # swap 2t 2j  ->  2j 2t

    # exp_data = A(0), Z(1) , N(2) , 2j(3) , 2t(4) , n(5) , energy(6) , error(7)
    # where energy(6) is absolute energy
    # THIS IS THE STANDARD COLUMN ORDER
    if use_pandas:
        data = pd.DataFrame(data,columns=['A', 'Z', 'N', '2j', '2t', 'n','energy','error'])

    return data

def get_df_from_csv(data_fn):
    df = pd.read_csv(data_fn)
    return df

#
#   SENSITIVITY ANALYSIS
#


def perturb_standard(pert_name,pert_vec,int_vec_file):
    """ perturbation to interaction matrix elements
     in standard basis (h.o. basis)
     this function not used much, replaced with perturbation in MILCOMs basis
     pert_name has the  form usdb_p<idx>_<dv>
    """
    remove_file(pert_name+".vec")
    remove_file(pert_name+".int")
    # PERTURB ORIGINAL VECTOR
    anamil_args = ["p",str(n_params),int_vec_file]
    for pert in pert_vec:
        anamil_args.append(str(int(pert[0]))+" "+str(pert[1]))
    anamil_args.append("0 0")
    anamil_args.append(pert_name+".vec")
    anamil_args.append("x")
    arun = subprocess.Popen(anamil_cmd,stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    arun.stdin.write("\n".join(anamil_args).encode())
    arun_out = arun.communicate()[0]
    #arun.wait()
    arun.stdin.close()
    if verbose: print("anamil args:\n"+"\n".join(anamil_args))
    if verbose: print(arun_out.decode('UTF-8'))

    # CONVERT TO INT
    standard_args = ["r",sps,pert_name,pert_name+".vec"]
    srun = subprocess.Popen(standard_cmd,stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    srun.stdin.write("\n".join(standard_args).encode())
    srun_out = srun.communicate()[0]
    #srun.wait()
    srun.stdin.close()
    if verbose: print("standardorderint args:\n"+"\n".join(standard_args))
    if verbose: print(srun_out.decode('UTF-8'))

def perturb_milcom(pert_name,pert_vec,original_mil_vec):
    """ mil_vec is interaction matrix elements in MILCOMs space
     aka PCA space

     original_mil_vec is interaction in MILCOMs space
     """
    pert_mil_name = pert_name[:4]+'mil'+pert_name[4:]
    # pert_name has the  form usdb_m<idx>_<dv>
    remove_file(pert_name+".vec")
    remove_file(pert_mil_name+".vec")
    remove_file(pert_name+".int")
    # PERTURB MILCOMS VECTOR
    pert_mil_args = ["p",str(n_params),original_mil_vec]
    for pert in pert_vec:
        pert_mil_args.append(str(int(pert[0]))+" "+str(pert[1]))
    pert_mil_args.append("0 0")
    pert_mil_args.append(pert_mil_name+".vec")
    pert_mil_args.append("x")
    pert_mil_run = subprocess.Popen(anamil_cmd,stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    pert_mil_run.stdin.write("\n".join(pert_mil_args).encode())
    pert_mil_run_out = pert_mil_run.communicate()[0]
    #pert_mil_run.wait()
    pert_mil_run.stdin.close()
    if verbose: print("perturb milcom args:\n"+"\n".join(pert_mil_args))
    if verbose: print(pert_mil_run_out.decode('UTF-8'))

    # TRANSFORM TO ORIGINAL BASIS
    if milcoms_filename.endswith('.milcom'):
        milcoms = milcoms_filename.replace('.milcom','')
    else:
        milcoms = milcoms_filename
    trans_mil_args = ["t",str(n_params),milcoms,pert_mil_name+'.vec','f'\
        ,pert_name+'.vec']
    trans_mil_args.append("x")
    trans_mil_run = subprocess.Popen(anamil_cmd,stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    trans_mil_run.stdin.write("\n".join(trans_mil_args).encode())
    trans_mil_run_out = trans_mil_run.communicate()[0]
    #trans_mil_run.wait()
    trans_mil_run.stdin.close()
    if verbose: print("transform milcom args:\n"+"\n".join(trans_mil_args))
    if verbose: print(trans_mil_run_out.decode('UTF-8'))

    # CONVERT TO INT
    standard_args = ["r",sps,pert_name,pert_name+".vec"]
    srun = subprocess.Popen(standard_cmd,stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    srun.stdin.write("\n".join(standard_args).encode())
    srun_out = srun.communicate()[0]
    #srun.wait()
    srun.stdin.close()
    if verbose: print("standardorderint args:\n"+"\n".join(standard_args))
    if verbose: print(srun_out.decode('UTF-8'))


#def find_state_index(wfn_initial,wfn_final,i_state):
#        overlaps = compute_overlaps(wfn_initial,wfn_final,i_state)
#        return np.argmax(overlaps[:,-1])+1

def match_states(current_data, spectrum, twoJz,fhlog):
    current_nstates = current_data.shape[0]

    #only keep states with 2*J = twoJz
    #spectrum = spectrum[ (2*spectrum['J']).astype(int) == twoJz]

    for i,current_data_state in enumerate(current_data):   #match states in current spectra
        for j,spectrum_state_tup in enumerate(spectrum):
            spectrum_state = np.array(list(spectrum_state_tup))
            current_data_state_2j2tn = np.array([float(x) for x in current_data_state[[3,4,5]]])
            current_data_state_2j2tn = current_data_state_2j2tn.astype(int)
            spectrum_state_2j2tn = np.array([float(x) for x in spectrum_state[[3,4,5]]])
            spectrum_state_2j2tn[[0,1]] = 2.0 * spectrum_state_2j2tn[[0,1]]
            spectrum_state_2j2tn = spectrum_state_2j2tn.astype(int)
            if (current_data_state_2j2tn == spectrum_state_2j2tn).all():
                if verbose: print("MATCHED")
                if verbose: print("exp:\t%i %i %i %i %i %i %f %f %f" % tuple(current_data[i]))
                if verbose: print("sm:\t"+" "+str(spectrum_state))
                fhlog.write("\nMATCHED \n")
                fhlog.write("exp:\t%i %i %i %i %i %i %f %f %f \n" % tuple(current_data[i]))
                fhlog.write("sm:\t"+str(spectrum_state)+"\n")

                current_data[i,-1] = spectrum_state[1]

                spectrum = np.delete(spectrum,(j),axis=0)
                break
    return current_data, spectrum


def compute_chi_squared(filename_int,use_exp_errors,theory_err):

    filename_in = filename_int +'_appended.dat'
    data = np.loadtxt(filename_in,delimiter="\t")
    print("DATA SHAPE =" + str(data.shape))
    if use_exp_errors == True:
        ndata = data.shape[0]
        exp_err_vec = np.sqrt( data[:,7]**2 + (theory_err*np.ones(ndata))**2 )
        residual = (data[:,-1] - data[:,6]) / exp_err_vec
    elif use_exp_errors == False:
        residual = (data[-1] - data[6])
    chisq = np.dot(residual,residual)

    if verbose: print("MAX ABS DIFF = "+str(max([abs(ed) for ed in residual])))
    if verbose: print("CHI SQUARED = "+str(chisq))
    return chisq

#
# EXPECTATION VALUES
#

def make_ops_vec(int_vec):
    """ "ops_vec" is just o(i) = delta(i) = 1 on i, 0 otherwise
     here, "operators" refer to the individual 1-body and 2-body density operators that make up H
     """
    for iop in range(n_params):
        basis_vec = np.zeros(n_params)
        basis_vec[iop] = 1
        outfn = int_name+'_o'+str(iop+1).zfill(3)+'.vec'
        np.savetxt(outfn,basis_vec,delimiter="\n",fmt="%f10",header=str(n_params),comments='')

def make_ops_int():
    for iop in range(n_params):
        op_name = int_name+'_o'+str(iop+1).zfill(3)
        op_name_vec = op_name+'.vec'
        op_name_int = op_name+'.int'
        if os.path.isfile(op_name_int):
            rm_cmd = "rm "+ op_name_int
            subprocess.Popen(rm_cmd,shell=True)
        # CONVERT TO INT
        standard_args = ["r",sps,op_name,op_name_vec]
        srun = subprocess.Popen(standard_cmd,stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        srun.stdin.write("\n".join(standard_args).encode())
        srun_out = srun.communicate()[0]
        srun.stdin.close()
        srun.wait()
        print("standardorderint args:\n"+"\n".join(standard_args))

def gather_exp_values(exp_data,fhout,op_name,make_log = False):
    """ Reads expectation values
    CAUTION: this reads from ALL .res files present with mathing int name
    """
    op_num = int(op_name[-3:])
    cases = glob.glob(f'*_{int_name}_o'+str(op_num).zfill(3)+'*.res')
    cases = sorted(cases,key=operator.itemgetter(1,2))
    if verbose: print(cases)
    if make_log:
        fhlog = open(op_name+"_gather.log",'w')
    else:
        fhlog = None
    match_list=[]
    for cn in cases:
        if make_log: fhlog.write("CASE "+cn+"\n")
        if verbose: print("CASE "+cn)
        Zi = int(cn[1:3])
        Ni = int(cn[4:6])
        spectrum = np.array(getspectrum(cn))
        spectrum = spectrum.astype(np.float)
        spectrum = count_duplicates(spectrum,match_columns=[2,3],round_columns=None)
        for ed in exp_data:
            found = False
            if ("".join(str(ed)) in match_list):
                continue
            if ed[1]==Zi and ed[2]==Ni:
                for sd in spectrum:
                    #match T^2 = T(T+1)
                    exp_jtn = [ed[3],ed[4],ed[5]]   # exp J T n
                    spec_jtn = [sd[2],sd[3],sd[6]]  #spectrum J T n
                    exp_jtn[0] = float(exp_jtn[0])/2 #exp J
                    exp_jtn[1] = (float(exp_jtn[1])/2)*((float(exp_jtn[1])/2) + 1) #exp T^2
                    if verbose:
                        print('exp_jtn',exp_jtn)
                        print('spec_jtn',spec_jtn)
                    if make_log: fhlog.write("CHECK jtn:\n"+str(exp_jtn)+"\n"+str(spec_jtn)+"\n")
                    if exp_jtn==spec_jtn:
                        if verbose: print(sd)
                        # outline = Z   N   J   T   n   Esm    Eexp     dEexp   op#     <o>
                        outline = [str(Zi),str(Ni)]+["%i" % (x) for x in spec_jtn]\
                            +["%f" % x for x in [sd[1],ed[6],ed[7]]]+[str(op_num),str(sd[4])]
                        outstring = "\t".join(outline) + "\n"
                        fhout.write(outstring)
                        #out_array.append(outline)
                        if make_log: fhlog.write("MATCH \n"+str(ed)+"\n"+str(sd)+"\n")
                        if make_log: fhlog.write("OUT\n"+outstring)
                        if verbose:
                            print('MATCH',ed,sd)
                            print('CASE',Zi,Ni)
                            print("OUT",outstring)
                        found = True
                        match_list.append("".join(str(ed)))
                        break
                    else:
                        if make_log: fhlog.write("NOPE \n"+str(ed)+"\n"+str(sd)+"\n")
                        #print('NO MATCH',ed,sd)
                if not found:
                    if make_log: fhlog.write("MISSING"+"\n"+str(ed)+"\n")

    #out_array =
    if make_log: fhlog.close()


#
# DEPRECATED FUNCTIONS
#  DO NOT USE THESE
#  Keeping them here for backward compatibility only
#

def compute_observables(filename_int,keep_wfn=False):
    """DEPRECATED
    Instead, use make_bigstick_input and run_bigstick
     used originally for USDB SA
    """
    error_states=[]
    if filename_int.endswith('.int'):
        filename_int = filename_int[:-4]
    fhlog = open(filename_int+".log",'w')
    cases=[]  # cases defined by Z,N for individual bigstick run
    for pair in [list(p) for p in list(brown_data[:,1:3])]:
        if pair not in cases:
            cases.append(pair)
    cases = np.array(cases)
    ndata = len(brown_data)
    #Ebigstick = np.zeros(ndata) # absolute energies from bigstick corresponding to brown_data
    case_data_list = []
    for ncase,ZNpair in enumerate(cases):
        #case_name = "Z"+str(int(ZNpair[0]))+"N"+str(int(ZNpair[1]))
        if verbose: print("COMPUTING CASE Z=%i N=%i" % tuple(ZNpair))
        fhlog.write("\nCOMPUTING CASE Z=%i N=%i \n" % tuple(ZNpair))

        current_data = [] #exp data for case = ZNpair
        for i,entry in enumerate(brown_data):
            if not (entry[1:3]-ZNpair).any():
                current_data.append(entry)
        current_data = np.array(current_data)
        fhlog.write("\nEXP DATA: \n")
        if verbose: print("EXP DATA:")
        for current_line in current_data:
            if verbose: print("%i %i %i %i %i %i %f %f" % tuple(current_line))
            fhlog.write("%i %i %i %i %i %i %f %f \n" % tuple(current_line))

        # add last column to current_data for bigstick energies
        current_data = np.hstack((current_data,np.zeros((current_data.shape[0],1))))

        debug = False     # BYPASS OBSERVABLE CALCULATION
        if not debug:
            twoJ_unique = np.unique(current_data[:,3]).astype(int)
            for twoJz in twoJ_unique:
                if any(current_data[current_data[:,3]==twoJz,-1]==0.0):
                    print("COMPUTING WITH twoJz = %i" % twoJz)
                else:
                    print("ALL STATES ACCOUNTED FOR WITH twoJz = %i" % twoJz)
                    continue
                spectrum = compute_bigstick_spectrum(ZNpair,twoJz,filename_int,keep_wfn,fhlog)

                if verbose: print("BIGSTICK SPECTRUM:")
                fhlog.write(" \nBIGSTICK SPECTRUM: \n")
                for spec_line in spectrum:
                    if verbose: print(spec_line)
                    fhlog.write(str(spec_line)+"\n")

                current_data, spectrum = match_states(current_data, spectrum, twoJz, fhlog)

            if not current_data[:,-1].all():
                print('MISSING STATE:')
                print(current_data)
                exit()

        case_data_list.append(current_data)

    output_data = np.vstack(case_data_list)
    #dtype_list = [('A','int'),('Z','int'),('N','int'),('2j','int'),('2t','int'),('n','int'),('Eexp','float'),('dEexp','float'),('Esm','float')]
    #output_data = np.array(output_data,dtype=dtype_list)
    fnout = filename_int +'_appended.dat'
    np.savetxt(fnout,output_data,fmt="\t".join(['%i']*6+['%f']*3))
    fhlog.close()


def parse_strength_file_gt(gt_run_name,state_index_list,dataframe):
    """ DEPRECATED! Use parse_strength_file_by_idx instead! """
    reading_M = False   # B = |M|^2 , are we reading B or M?
    if reading_M:
        fn = gt_run_name+'.mstr'  #filename convention
    elif not reading_M:
        fn = gt_run_name+'.str'  #filename convention
    if verbose: print("Parsing "+fn+"...")
    count=0
    for i_t,t in dataframe.iterrows():
        i_parent = state_index_list[count][0]
        i_daughter = state_index_list[count][1]
        count+=1
        with open(fn,'r') as fh:
            if verbose: print(f'i_t = {i_t}')
            line_num = 0
            if verbose:print('transition:',t)
            found_parent = False
            found_daughter = False
            for line in fh:
                line_num += 1
                if line_num < 3:
                    continue
                ls = line.split()
                if found_parent and ('parent' in ls):
                    print(t)
                    sys.exit(f'ERROR: initial state has no final state: {gt_run_name}')
                if ('parent' in ls) and (int(ls[0])==int(i_parent+1)):
                    found_parent = True
                    if verbose: print("found initial state:",line)
                if found_parent and ('daughter' in ls) and (int(ls[0])==int(i_daughter+1)):
                    found_daughter = True
                    dataframe.loc[i_t,'Bth'] = float(ls[4])
                    if verbose: print("found final state:",line)
                    break
            if not found_parent:
                print(t)
                sys.exit(f'ERROR: initial state is missing: {gt_run_name}')
            if not found_daughter:
                print(t)
                sys.exit(f'ERROR: final state is missing: {gt_run_name}')
    return dataframe





