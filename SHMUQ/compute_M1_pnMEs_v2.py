#
#   computes many samples of M1 transitions using BIGSTICK
#   Fox 2021
#
from shmuq import *

debug = False
if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_filename_csv',type=str,default='sd_M1_processed.csv', help='csv to read from')
    parser.add_argument('-o','--output_filename_csv',type=str,default='sd_M1_complete.csv', help='csv to write to')
    parser.add_argument('-s','--sample_number',type=int, default=None, help='number to tag random interaction')
    parser.add_argument('--skip_bigstick_calc',action='store_true', help='skip BIGSTICK calculations')
    parser.add_argument('--skip_strength_calc',action='store_true', help='skip genstrength calculations')

    args = parser.parse_args()

    do_bigstick_calc = not args.skip_bigstick_calc 
    do_strength_calc = not args.skip_strength_calc 
    sample_number = args.sample_number
    if sample_number is None:
        sample_int_name = int_name
    else:
        sample_int_name = make_sample_interaction(int_name,sample_number,milcoms_filename)

    df = pd.read_csv(args.input_filename_csv)

    #exclude transitions with too high excitations, or too small strength
    max_n = 6  
    B_cutoff = 0.01
    df.loc[ df[ df['B_exp (W.u.)']<=B_cutoff ].index , 'Include' ] = False
    df.loc[ df[ df['ni']>max_n ].index , 'Include' ] = False
    df.loc[ df[ df['nf']>max_n ].index , 'Include' ] = False
    
    df = df[df['Include']==True]
    
    df['Mth_sp'] = -999.0
    df['Mth_sn'] = -999.0
    df['Mth_lp'] = -999.0
    df['Mth_ln'] = -999.0

    nuclei = np.array(df[['A','Element','Z','N']].drop_duplicates())

    for nuc_entry in nuclei:
        if verbose: print('nuc_entry',nuc_entry)
        transitions = df[(df['A']==nuc_entry[0]) & (df['Element']==nuc_entry[1])]   #'transitions' is a subset of dataframe

        nuc = get_nucleus(transitions.iloc[0])

        twoJz = nuc.twoJz
        run_name_primary = make_bigstick_input(sample_int_name,nuc,'d',twoJz=twoJz)
        if do_bigstick_calc: run_bigstick(run_name_primary)
        density_list = [run_name_primary]
        if twoJz==0:
            # run_name_b only used when CG could be zero
            run_name_secondary = make_bigstick_input(sample_int_name,nuc,'d',twoJz=2)
            if do_bigstick_calc: run_bigstick(run_name_secondary)
            density_list.append(run_name_secondary)

        run_name_gs_sp = make_genstrength_inputs_v2(run_name_primary,sample_int_name,nuc,op_name['M1_sp'],density_list)
        run_name_gs_sn = make_genstrength_inputs_v2(run_name_primary,sample_int_name,nuc,op_name['M1_sn'],density_list)
        run_name_gs_lp = make_genstrength_inputs_v2(run_name_primary,sample_int_name,nuc,op_name['M1_lp'],density_list)
        run_name_gs_ln = make_genstrength_inputs_v2(run_name_primary,sample_int_name,nuc,op_name['M1_ln'],density_list)
        if do_strength_calc: 
            run_genstrength(run_name_gs_sp)
            run_genstrength(run_name_gs_sn)
            run_genstrength(run_name_gs_lp)
            run_genstrength(run_name_gs_ln)
        
        if sample_number==None:
            if (nuc.Zv>0) and ((max_Z - core_Z - nuc.Zv) > 0):
                if verbose: print('Nonzero proton...')
                if verbose: print('Parsing strength file for sp...')
                transitions = parse_strength_file(run_name_gs_sp,transitions,'Mth_sp', reading_m=True)
                if debug: breakpoint()
                if verbose: print('Parsing strength file for lp...')
                transitions = parse_strength_file(run_name_gs_lp,transitions,'Mth_lp', reading_m=True)
                if debug: breakpoint()
            else:
                if verbose: print('NO PROTON PART!')
                transitions['Mth_sp'] = 0.
                transitions['Mth_lp'] = 0.

            if (nuc.Nv>0) and ((max_N - core_N - nuc.Nv) > 0):
                if verbose: print('Nonzero neutron...')
                if verbose: print('Parsing strength file for sn...')
                transitions = parse_strength_file(run_name_gs_sn,transitions,'Mth_sn', reading_m=True)
                if debug: breakpoint()
                if verbose: print('Parsing strength file for ln...')
                transitions = parse_strength_file(run_name_gs_ln,transitions,'Mth_ln', reading_m=True)
                if debug: breakpoint()
            else:
                if verbose: print('NO NEUTRON PART!')
                transitions['Mth_sn'] = 0.
                transitions['Mth_ln'] = 0.
        else:
            # this is run when randomly sampling interaction
            state_index_list = state_indices_pairs_from_ovlp(transitions,sample_int_name,run_name_primary,unperturbed_results_dir_M1)
            print('State index list',state_index_list)
            if (nuc.Zv>0) and ((max_Z - core_Z - nuc.Zv) > 0):
                if verbose: print('Nonzero proton...')
                if verbose: print('Parsing strength file for sp...')
                transitions = parse_strength_file_by_idx(run_name_gs_sp,state_index_list,transitions,'Mth_sp', reading_m=True)
                if verbose: print('Parsing strength file for lp...')
                transitions = parse_strength_file_by_idx(run_name_gs_lp,state_index_list,transitions,'Mth_lp', reading_m=True)
            else:
                if verbose: print('NO PROTON PART!')
                transitions['Mth_sp'] = 0.
                transitions['Mth_lp'] = 0.

            if (nuc.Nv>0) and ((max_N - core_N - nuc.Nv) > 0):
                if verbose: print('Nonzero neutron...')
                if verbose: print('Parsing strength file for sn...')
                transitions = parse_strength_file_by_idx(run_name_gs_sn,state_index_list,transitions,'Mth_sn', reading_m=True)
                if verbose: print('Parsing strength file for ln...')
                transitions = parse_strength_file_by_idx(run_name_gs_ln,state_index_list,transitions,'Mth_ln', reading_m=True)
            else:
                if verbose: print('NO NEUTRON PART!')
                transitions['Mth_sn'] = 0.
                transitions['Mth_ln'] = 0.

        if verbose: print('TRANSITION DATA FOR THIS NUCLEUS:',transitions)
        # setting elements from trasnsitions df to main df
        df.loc[(df['A']==nuc_entry[0]) & (df['Element']==nuc_entry[1]), 'Mth_sp'] = transitions['Mth_sp']
        df.loc[(df['A']==nuc_entry[0]) & (df['Element']==nuc_entry[1]), 'Mth_lp'] = transitions['Mth_lp']
        df.loc[(df['A']==nuc_entry[0]) & (df['Element']==nuc_entry[1]), 'Mth_sn'] = transitions['Mth_sn']
        df.loc[(df['A']==nuc_entry[0]) & (df['Element']==nuc_entry[1]), 'Mth_ln'] = transitions['Mth_ln']

    df.to_csv(args.output_filename_csv)

    print("Done." + f'Results written to {args.output_filename_csv}')




