import sys
from os.path import realpath, dirname
sys.path.insert(0, dirname(realpath('')))

import numpy as np
import pandas as pd
import scipy.interpolate
import warnings
import time
import pylandau
from pylandau import langau
from util.cer_util import CER
from util.theory import langau_pdf
from tqdm.auto import tqdm
warnings.filterwarnings('ignore')
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Analyze import initialize, load_slimming_data, load_data, delta_rm, display_uptime


def get_inputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--save", default='', help="Save to file in \'./data/reconstructions\'")
    parser.add_argument("-f", "--fitloc", default='base_fit_data.csv', help="Read fit information from \'./data/fit_data\'")
    parser.add_argument("-c", "--cut", default=False, nargs=2, type=float, help="Cut dedx < 1.25 MeV and dedx > 6 MeV")
    parser.add_argument("-p", "--pitch-lims", default=[0.3,0.4], nargs=2, type=float, help="Limit pitch between pitch-lims in cm (used in conjunction with pitch-limited fitloc)")
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


def slim_further(df, part_df, pitch_lims):
    # Final condition is always true, just makes sure pitch_mask has the right shape
    pitch_mask = (0.3 <= df.pitch_y.loc[:,0]) & (df.pitch_y.loc[:,0] <= 0.4) & (part_df.backtracked_e != 0)
    mi_pitch_mask = pitch_mask[df.index.get_level_values(0)].to_numpy()
    
    part_df = part_df.loc[pitch_mask]
    df = df.loc[mi_pitch_mask, :]
    
    print('Initial Pitch-Slimmed Size:', sys.getsizeof(df)/1e6, 'MB')
    
    return df, part_df


def truncate(df, pitch_lims):
    df = df.droplevel(level=0)
    bad_indices = df.index[(df.dedx_y > 100) | (df.pitch_y < pitch_lims[0]) | (df.pitch_y > pitch_lims[1])]
    if len(bad_indices) == 0:
        return df
    
    trunc = min(bad_indices)
    if trunc < 10:
        return df.iloc[:0]
    
    return df.iloc[:trunc]
    
    
def reconstruct_e(df, pdfs, lookup_df, index):
    df = df.droplevel(level=0)
    
    dedxs = df.dedx_y.to_numpy().astype(np.float64)
    
    lognorm = np.log(np.sum(np.array([ pdf(dedxs) for pdf in pdfs ]), axis=0))
    logmodel = np.log(np.array([ pdf(dedxs) for pdf in pdfs ]))
    loglike = np.sum(logmodel-lognorm, axis=1)
    jtilde = np.argmax(loglike)
    
    e_min_tilde, e_max_tilde = lookup_df.iloc[jtilde]
    
    res = pd.Series([e_min_tilde, e_max_tilde, dedxs.shape[0], *loglike], index=index)
    return res

    
# Legacy reconstruction algorithm (slow)
def _reconstruct_e(df, l_params_matrix, lookup_df, index):
    df = df.droplevel(level=0)
    
    dedxs = df.dedx_y.to_numpy().astype(np.float64)
    
    lognorm = np.log(np.sum(np.array([ langau_pdf(dedxs, *params) for params in l_params_matrix ]), axis=0))
    logmodel = np.log(np.array([ langau_pdf(dedxs, *params) for params in l_params_matrix ]))
    loglike = np.sum(logmodel-lognorm, axis=1)
    jtilde = np.argmax(loglike)
    
    e_min_tilde, e_max_tilde = lookup_df.iloc[jtilde]
    
    res = pd.Series([e_min_tilde, e_max_tilde, dedxs.shape[0], *loglike], index=index)
    return res


def preprocess(df, pitch_lims, drm):
    start = time.perf_counter()
    print("Truncating...", end='')
    data = df.groupby(level=0).apply(truncate, pitch_lims)
    display_uptime(start)
    
    if drm != [0,0]:
        print("Removing Delta-Rays...", end='')
        data = data.groupby(level=0).apply(delta_rm, *drm)
        display_uptime(start)
    
    return data

# Test this
def generate_interpolated_pdfs(fitdata):
    x = np.append(np.linspace(0,10,1000),np.linspace(11,1000,10))
    pdfs = (
    fitdata.groupby(level=0)
    .apply(
            lambda df: scipy.interpolate.interp1d(x, langau_pdf(x, *df['mpv'], *df['eta'], *df['sigma']))
          )
    .to_list()
    )
    return pdfs
    
def reconstruct(data, fitdata):
    pdfs = generate_interpolated_pdfs(fitdata)
    start = time.perf_counter()
    num_bins = fitdata.shape[0]
    col_names = ['reconstructed_min', 'reconstructed_max', 'track_length', *(np.char.array(['L']*num_bins) + np.char.array(np.arange(num_bins)).astype(str))]
    lookup_df = fitdata[['e_min', 'e_max']]
    tqdm.pandas(desc="Reconstructing Energy", unit="muon")
    reconstruction_data = (
                           data.groupby(level=0)
                               .progress_apply(reconstruct_e, 
                                               pdfs=pdfs, 
                                               lookup_df=lookup_df, 
                                               index=col_names,
                                               )
                          )
    
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
    
    preprocessed_df = preprocess(df, pitch_lims, drm)
    
    result = reconstruct(preprocessed_df, fitdata)
    result = result.join(part_df.backtracked_e, on='entry')[['backtracked_e', *list(result.columns.values)]]
    result = result.rename(columns={'backtracked_e': 'truth'})
        
    if save:
        print(f"Saving to {savefile}...")
        result.to_csv(savefile, index=True, header=True)
        print("Saved!")
        
    display_uptime(start, "Complete! Total Uptime:")
    

if __name__ == "__main__": 
    main()