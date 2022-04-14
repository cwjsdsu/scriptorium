#
#   computes many samples of Gamow-Teller transitions using BIGSTICK
#   Fox 2019
#

from shmuq import *

match_by_ovlp = False # if true, compute ovlps, if false use J_n index

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--input_filename_csv',type=str,default='sd_GT_processed.csv', help='csv to read from')
    parser.add_argument('-o','--output_filename_csv',type=str,default='sd_GT_complete.csv', help='csv to write to')
    parser.add_argument('-s','--sample_number',type=int, default=None, help='number to tag random interaction')
    parser.add_argument('--skip_bigstick_calc',action='store_true', help='skip BIGSTICK calculations, but do everything else')

    args = parser.parse_args()

    do_bigstick_calc = not args.skip_bigstick_calc 
    sample_number = args.sample_number
    if sample_number is None:
        sample_int_name = int_name
    else:
        sample_int_name = make_sample_interaction(int_name,sample_number,milcoms_filename)

    df = pd.read_csv(args.input_filename_csv)
    df['Bth'] = 0.0
    df = df[df['include']==True]

    unique_pairs = pd.unique(list(zip(df['parent'],df['daughter'])))

    for pair in unique_pairs:
        transitions = df[(df['parent']==pair[0]) & (df['daughter']==pair[1])]   #'transitions' is a subset of dataframe
        parent = get_parent(transitions.iloc[0])
        daughter = get_daughter(transitions.iloc[0])
        twoJz_min = np.min(list(zip(transitions['twoJi'],transitions['twoJf'])))
        if verbose: print(f'minimum(twoJz) = {twoJz_min}')
        Tmirror = transitions.iloc[0]['Tmirror']

        if Tmirror:
            Tshift = 0
        else:
            Tshift = 2

        run_name_par = None
        run_name_dau = None
        run_name_den = None
        run_name_den_2 = None
        density_list=[]
        if not Tmirror:
            run_name_par = make_bigstick_input(sample_int_name,parent,'n',twoJz=twoJz_min,Tshift=0)
            run_name_dau = make_bigstick_input(sample_int_name,daughter,'n',twoJz=twoJz_min,Tshift=0)

            if abs(parent.twoTz)>abs(daughter.twoTz):
                if verbose: print(f"Tshift={Tshift}. Densities are computed in the daughter space")
                run_name_den = make_bigstick_input(sample_int_name,daughter,'d',twoJz=twoJz_min,Tshift=Tshift)
                if twoJz_min==0:
                    # compute another density file with J=1
                    run_name_den_2 = make_bigstick_input(sample_int_name,daughter,'d',twoJz=2,Tshift=Tshift)

            else:
                if verbose: print(f"Tshift={Tshift}. Densities are computed in the parent space")
                run_name_den = make_bigstick_input(sample_int_name,parent,'d',twoJz=twoJz_min,Tshift=Tshift)
                if twoJz_min==0:
                    run_name_den_2 = make_bigstick_input(sample_int_name,parent,'d',twoJz=2,Tshift=Tshift)

            if do_bigstick_calc:
                run_bigstick(run_name_par)
                run_bigstick(run_name_dau)
                run_bigstick(run_name_den)
                if twoJz_min==0:
                    run_bigstick(run_name_den_2)

        else:
            if verbose: print('Isospin mirror')
            if verbose: print(f"Tshift={Tshift}. Densities are computed in the parent space")
            run_name_par = make_bigstick_input(sample_int_name,parent,'d',twoJz=twoJz_min,Tshift=Tshift)
            run_name_dau = run_name_par
            run_name_den = run_name_par
            if twoJz_min==0:
                run_name_den_2 = make_bigstick_input(sample_int_name,parent,'d',twoJz=2,Tshift=Tshift)
            if do_bigstick_calc:
                run_bigstick(run_name_par)
                if twoJz_min==0:
                    run_bigstick(run_name_den_2)

        density_list = [run_name_den]
        if twoJz_min==0:
            density_list.append(run_name_den_2)
        run_name_gt = make_gtstrength_inputs(run_name_par,run_name_dau,density_list,parent,daughter,Tshift)
        run_gtstrength(run_name_gt)
        if sample_number is None: # then do not look for USDB runs as reference
            transitions = parse_strength_file(run_name_gt,transitions,'Bth',skip_lines=2)
        else:
            if match_by_ovlp:
                state_idx_list = parent_daughter_indices(transitions,sample_int_name,run_name_par,run_name_dau, unperturbed_results_dir_GT)
                if verbose: print('State idx list =',state_idx_list)
                transitions = parse_strength_file_by_idx(run_name_gt,state_idx_list,transitions,'Bth', skip_lines=2)
            else:
                transitions = parse_strength_file(run_name_gt,transitions,'Bth',skip_lines=2)


        # put strength values in df
        print('TRANSITIONS ARRAY:')
        print(transitions)
        print('SUBSET ARRAY:')
        print(df.loc[(df['parent']==pair[0]) & (df['daughter']==pair[1])])
        df.loc[(df['parent']==pair[0]) & (df['daughter']==pair[1]),'Bth'] = transitions['Bth']

    df.to_csv(args.output_filename_csv)
    if sample_int_name != 'usdb':
        move_files_to_folder(sample_int_name,sample_number)

    print("Done")




