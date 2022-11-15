import sys
from os.path import realpath, dirname
sys.path.insert(0, dirname(realpath('')))

import numpy as np
import pandas as pd
import warnings
import time
import pylandau
from pylandau import langau
from util.cer_util import CER
warnings.filterwarnings('ignore')
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import uproot
from Analyze import initialize, load_slimming_data, load_data, delta_rm, display_uptime


def get_inputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--save", default='', help="Save to file in \'./data/reconstructions\'")
    parser.add_argument("-f", "--fitloc", default='base_fit_data.csv', help="Read fit information from \'./data/fit_data\'")
    parser.add_argument("-c", "--cut", default=False, nargs=2, type=float, help="Cut dedx < 1.25 MeV and dedx > 6 MeV")
    parser.add_argument("-p", "--pitch-lims", default=[0.3,0.87714132], nargs=2, type=float, help="Limit pitch between pitch-lims in cm (used in conjunction with pitch-limited fitloc)")
    parser.add_argument("-e", "--energy-lims", default=[0.1,100], nargs=2, type=float, help="Limit energy between energy-lims in GeV (used in conjunction with energy-limited fitloc)")
    parser.add_argument("-d", "--delta-rm-params", default=[0,0], nargs=2, type=int, help="num_sig and buff for delta-ray removal")
    parser.add_argument("--full", default=False, action='store_true', help='Load the full dataset (may take a long time)')
    args = vars(parser.parse_args())
    

    save = args['save']
    fit_data_loc = rf"../data/fit_data/{args['fitloc']}"
    cut = args['cut']
    pitch_lims = args['pitch_lims']
    e_lims = args['energy_lims']
    full = args['full']
    drm = args['delta_rm_params']
    
    return save, fit_data_loc, cut, pitch_lims, e_lims, full, drm

# Vectorized langau_pdf calculation
def langau_pdf(dedxs, params):
    sf = 100
    scaled_dedxs = dedxs*sf
    scaled_params = params*sf
    return sf * scaled_params[1] * pylandau.langau_pdf(scaled_dedxs, *scaled_params)


def slim_further(df, part_df, pitch_lims):
    # Final condition is always true, just makes sure pitch_mask has the right shape
    pitch_mask = (0.3 <= df.pitch_y.loc[:,0]) & (df.pitch_y.loc[:,0] <= 0.4) & (part_df.backtracked_e != 0)
    mi_pitch_mask = pitch_mask[df.index.get_level_values(0)].to_numpy()
    
    part_df = part_df.loc[pitch_mask]
    df = df.loc[mi_pitch_mask, :]
    
    print('Initial Pitch-Slimmed Size:', sys.getsizeof(df)/1e6, 'MB')
    
    return df, part_df


def like_max(dedxs, fitdata):
    # if cut:
    #     dedxs = dedxs[(dedxs > cut[0]) & (dedxs < cut[1])]
    
    landau_params = np.array([ fitdata[['mpv', 'eta', 'sigma']].iloc[i] for i in range(fitdata.shape[0]) ])
    
    # One big list comprehension for maximum calculation speed
    loglike = np.array([ np.sum([ np.log(langau_pdf(xi, *fj_params)) - np.log(np.sum([ langau_pdf(xi, *fk_params) for fk_params in landau_params])) for xi in dedxs ]) for fj_params in landau_params])
    
    jtilde = np.argmax(loglike)
    e_min_tilde, e_max_tilde = fitdata[['e_min', 'e_max']].iloc[jtilde]
    return e_min_tilde, e_max_tilde, loglike


def truncate(df, pitch_lims):
    df = df.droplevel(level=0)
    bad_indices = df.index[(df.dedx_y > 100) | (df.pitch_y < pitch_lims[0]) | (df.pitch_y > pitch_lims[1])]
    if len(bad_indices) == 0:
        return df
    
    trunc = min(bad_indices)
    if trunc < 10:
        return df.iloc[:0]
    
    return df.iloc[:trunc]
    

def reconstruct_e(df, l_params_matrix, lookup_df, index):
    df = df.droplevel(level=0)
    
    dedxs = df.dedx_y.to_numpy().astype(np.float64)
    
    lognorm = np.log(np.sum(np.array([ langau_pdf(dedxs, params) for params in l_params_matrix ]), axis=0))
    logmodel = np.array([ langau_pdf(dedxs, params) for params in l_params_matrix ])
    loglike = np.sum(logmodel-lognorm, axis=1)
    jtilde = np.argmax(loglike)
    
    e_min_tilde, e_max_tilde = lookup_df.iloc[jtilde]
    
    res = pd.Series([e_min_tilde, e_max_tilde, *loglike], index=index)
    return res


def reconstruct(df, fitdata, pitch_lims, drm):
    start = time.perf_counter()
    print("Truncating...", end='')
    data = df.groupby(level=0).apply(truncate, pitch_lims)
    display_uptime(start)
    
    if drm != [0,0]:
        print("Removing Delta-Rays...", end='')
        data = data.groupby(level=0).apply(delta_rm, *drm)
        display_uptime(start)
    
    print("Reconstructing Energy...", end='')
    col_names = ['reconstructed_min', 'reconstructed_max', *(np.char.array(['L']*10) + np.char.array(np.arange(10)).astype(str))]
    l_params = fitdata[['mpv', 'eta', 'sigma']].to_numpy()
    lookup_df = fitdata[['e_min', 'e_max']]
    reconstruction_data = data.groupby(level=0).apply(reconstruct_e, l_params, lookup_df, col_names)
    
    display_uptime(start, 'Done! Total Reconstruction Time:')
    return reconstruction_data

    
def main():
    start = time.perf_counter()
    save, fit_data_loc, cut, pitch_lims, e_lims, full, drm = get_inputs()
    
    if cut:
        print("Warning: dE/dx cuts not currently supported")
    
    if save:
        save = r'reconstructions/' + save
        
    tree, savefile = initialize(full, save)
    mask = load_slimming_data(tree, e_lims)
    df, part_df = load_data(tree, mask)
    df, part_df = slim_further(df, part_df, pitch_lims)
    fitdata = pd.read_csv(fit_data_loc)
    
    result = reconstruct(df, fitdata, pitch_lims, drm)
    result = result.join(part_df.backtracked_e, on='entry')[['backtracked_e', *list(result.columns.values)]]
    result = result.rename(columns={'backtracked_e': 'truth'})
    
    if save:
        print(f"Saving to {savefile}...")
        result.to_csv(savefile, index=False, header=True)
        print("Saved!")
        
    display_uptime(start, "Complete! Total Uptime:")
    

if __name__ == "__main__": 
    main()