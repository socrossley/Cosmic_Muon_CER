import sys
from os.path import realpath, dirname
sys.path.insert(0, dirname(realpath('')))

import pandas as pd
import numpy as np
import util.theory as theory
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import fsolve, curve_fit
import scipy.stats
from tqdm.auto import tqdm
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
import reconstruct
from landaupy import langauss
warnings.filterwarnings('ignore')

def get_inputs():
    # Parse argments inputted on the command line
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--save-reconstruction", default='', help="Save to file in \'./data/reconstructions\'")
    parser.add_argument("-m", "--save-muon-tracks", default='', help="Save to file in \'./data\'")
    parser.add_argument("-n", "--num-per-ebin", default='200', type=int, help="How many MC muons to reconstruct per energy bin")
    parser.add_argument("--mc-only", default=False, action='store_true', help="Produce the MC muon dedxs only (no reconstruction)")
    parser.add_argument("-b", "--bias", default=0, type=float, help="Bias percentage to introduce to the generated dE/dx")
    parser.add_argument("-e", "--error", default=0, type=float, help="Error percentage to introduce to the generated dE/dx")
    parser.add_argument("-f", "--fitloc", default='narrow_lowpitch_fixedsig_fit_data.csv', 
                        help="Location in './data/fit_data/' of Langau models from which to generate MC dE/dx values.")
    parser.add_argument("--energy-range", default=(1,10), type=(float, float), help="Energy range over which to generate MC muons.")
    
    args = vars(parser.parse_args())

    return args


def initialize(fits, spread_pct):    
    fitdf = pd.read_csv('../data/fit_data/' + fits)

    langau_params = fitdf[['mpv', 'eta', 'sigma']]
    e_bins = fitdf[['e_min', 'e_max']]
    return langau_params, e_bins


# Vectorized!
def sample_from_langau(mpv, eta, sigma, size):
    # Samples according to landaupy, correction to mpv must be made first
    mpv -= theory.mpv_conv * eta
    return langauss.sample(mpv, eta, sigma, size)
                        
    
def rand_trkl(num_muons):
    # Generate a random track length from lognormal distribution that loosely follows actual track length distribution
    # Hardcoded track length distribution parameters
    s, loc, scale = ( 2.01980595e-01, -1.23848172e+03, 1.66933758e+03 )
    
    rand_trkls = np.zeros(num_muons)
    minlength, maxlength = 2, 3000
    bad = (rand_trkls < minlength) | (rand_trkls > maxlength)
    
    while bad.any():
        rand_trkls[bad] = scipy.stats.lognorm.rvs(s, loc, scale, np.sum(bad))
        bad = (rand_trkls < minlength) | (rand_trkls > maxlength)
    
    return rand_trkls.astype(int)


def display_uptime(start, msg=''):
    now = time.perf_counter()
    t = now-start
    print(f'{msg} {int(t//60)}m {t%60:0.1f}s')
    return now


def generate_dedxs(df, rng):
    mpv, eta, sigma, *trkls = df.values[0]
    trkls = np.asarray(trkls, dtype=int)
    tot_trkls = int(trkls.sum())
    dedxs = sample_from_langau(mpv, eta, sigma, tot_trkls)
    track_lengths_per_dedx = np.repeat(trkls, trkls)
    
    num_muons = len(trkls)
    lvl0_offset = df.index[0]*num_muons
    entry = np.repeat(np.arange(num_muons), trkls) + lvl0_offset 
    # https://codereview.stackexchange.com/questions/83018/vectorized-numpy-version-of-arange-with-multiple-start-stop
    subentry = np.repeat(trkls - trkls.cumsum(), trkls) + np.arange(trkls.sum()) 
    index = pd.MultiIndex.from_arrays([entry, subentry], names=['entry','subentry'])
    return pd.DataFrame(np.transpose([dedxs, track_lengths_per_dedx]), index=index, 
                        columns=['dedx_y', 'track_length'])


# Perform the MC generation
def mc_generate(langau_params, e_bins, muons_per_ebin, energy_range):
    rng = np.random.default_rng()
    langau_params_for_generation = langau_params.loc[(e_bins['e_max'] > energy_range[0]) & (e_bins['e_min'] < energy_range[1])]
    num_bins = langau_params_for_generation.shape[0]
    tot_num_muons = muons_per_ebin * num_bins
    trkls = rand_trkl(tot_num_muons)
    
    generator_frame = langau_params_for_generation.copy()
    trkls_df = pd.DataFrame(trkls.reshape(num_bins, muons_per_ebin))
    generator_frame = generator_frame.join(trkls_df)
    tqdm.pandas(desc='Generating MC muon data', unit="bin")
    df = generator_frame.groupby(level=0).progress_apply(generate_dedxs, rng=rng)
    df = df.droplevel(0)
    df = df.astype({'track_length': 'uint8'})
    return df

    
def main():
    start = time.perf_counter()
    args = get_inputs()
    langau_params, e_bins = initialize(args['fitloc'], args['error'])
    muons_per_ebin = args['num_per_ebin']
    energy_range = args['energy_range']
    
    dedxs_df = mc_generate(langau_params, e_bins, muons_per_ebin, energy_range)
    display_uptime(start, "Complete:")
    
    dedxs_df.dedx_y += dedxs_df.dedx_y * args['bias']/100
    dedxs_df.dedx_y += scipy.stats.norm.rvs(0, dedxs_df.dedx_y*args['error']/100)
    
    dedxs_savefile = args['save_muon_tracks']
    rec_savefile = args['save_reconstruction']
    
    if dedxs_savefile:
        dedxs_df.dedx_y.to_csv(r'../data/' + dedxs_savefile)
        print("Saved MC-generated dE/dx data to \'./data/" + dedxs_savefile)
        
    
    result = reconstruct.reconstruct(dedxs_df, langau_params.join(e_bins))
    truth = e_bins.mean(axis=1).repeat(muons_per_ebin).rename('truth').reset_index().rename(columns={'index':'truebin'})
    result = result.join(truth)
    
    if rec_savefile:
        print("Saving Reconstruction Data to \'./data/reconstructions/" + rec_savefile + '...', end='')
        result.to_csv(r'../data/reconstructions/' + rec_savefile, index=False, header=True)
    
    display_uptime(start, msg="Complete! Total Uptime:")
    
    
if __name__ == '__main__':
    main()