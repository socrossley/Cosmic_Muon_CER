import sys
from os.path import realpath, dirname
sys.path.insert(1, dirname(realpath('')))
sys.path.insert(2, dirname(realpath(''))+'/util')
path = dirname(realpath(''))

import numpy as np
import pandas as pd
import os
import sys
import datetime
import gc
import warnings
import time
from theory import deltas, Mmu 
warnings.filterwarnings('ignore')
import csv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import uproot


def get_inputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-f", "--full", default=False, action='store_true', help="Use the full dataset")
    parser.add_argument("-s", "--save", default='', help="Save to file in \'" + path + r"/data/\'")
    args = vars(parser.parse_args())

    full = args['full']
    save = args['save']
    return full, save


def initialize(full, save):
    data_loc = path + r"/data/simulated_cosmics.root:CalorimetryAnalyzer"
    if full:
        data_loc = path + r"/data/simulated_cosmics_full.root:nuselection/CalorimetryAnalyzer"

    print("Using data location:", data_loc)

    savefile = path + r'/data/' + save
    if save: 
        print(f'Will save to {savefile}')
        
    tree = uproot.open(data_loc)
    return tree, savefile


def load_slimming_data(tree, elims=(0,100)):
    slimmer_variables = ['trk_sce_start_x','trk_sce_start_y','trk_sce_start_z', 'trk_sce_end_x','trk_sce_end_y','trk_sce_end_z','backtracked_e', 'backtracked_pdg']

    thresh = 2 #cm 
    def distance_to_edge(r):
        dimensions = np.array([[0, 256], [-116,116], [0,1036]])
        return  np.min(np.abs(dimensions - r[:, np.newaxis]))

    print("Preparing Slimming Mask...")
    slimmerdf = tree.arrays(slimmer_variables, library='pd')
    start_dists, end_dists = np.array([ [distance_to_edge(r[:3]), distance_to_edge(r[3:6])] for _, r in slimmerdf.iterrows() ]).T
    energy_mask = (slimmerdf.backtracked_e > elims[0]-Mmu/1000) & (slimmerdf.backtracked_e < elims[1]-Mmu/1000) & (np.abs(slimmerdf.backtracked_pdg) == 13)
    mask = ((start_dists < thresh) & (end_dists < thresh) & energy_mask).to_numpy()
    print("Will remove", np.sum(~mask), "particles")
    return mask


def load_data(tree, mask):
    per_particle_variables = ['backtracked_e','backtracked_pdg','backtracked_purity']
    variables = ['dedx_y','rr_y','pitch_y']
    
    print("Generating Principal Dataframe...")
    part_df = tree.arrays(per_particle_variables, library='pd')
    df = tree.arrays(variables[0], library='pd')
    print("Loaded", variables[0], "data...")
    size = sys.getsizeof(df)

    # Slim according to mask
    part_df = part_df[mask]
    mask = mask[df.index.get_level_values(0)] # Broadcast to multiindex shape
    df = df.loc[mask, :]

    # This loop loads in the next column of the dataframe, slims it, and appends it to df
    for name in variables[1:]:
        next_col = tree.arrays(name, library='pd')
        print("Loaded", name, "data...")
        size += sys.getsizeof(next_col[name])
        next_col = next_col.loc[mask, :]
        df = df.join(next_col, on=['entry', 'subentry'])

    part_df.index.name = 'entry'
    print("Generated!")
    print("Original Size:", size/10**6, "MB")
    print("Slimmed Size:", sys.getsizeof(df)/10**6, "MB")
    
    return df, part_df


# Begin Analysis


def truncate(df):
    df = df.droplevel(level=0)
    bad_indices = df.index[(df.dedx_y > 100) | (df.e_y <= 0) | (df.pitch_y < 0.3) | (df.pitch_y > 0.3/np.cos(70*np.pi/180))]
    if len(bad_indices) == 0:
        return df
    
    trunc = min(bad_indices)
    if trunc < 10:
        return df.iloc[:0]
    
    return df.iloc[:trunc]

def delta_rm(df, num_sig=3, buff=3):
    df = df.droplevel(level=0)
    
    delta_locs, count = deltas(df.dedx_y.to_numpy(), num_sig=num_sig, buff=buff)
    return df.drop(delta_locs, axis=0)    

def display_uptime(start, msg=''):
    now = time.perf_counter()
    t = now-start
    print(f'{msg} {int(t//60)}m {t%60:0.1f}s')
    return now
    
    
# Principal Analysis Function
def analyze_data(df, part_df):

    start = time.perf_counter()
    print('Analyzing...', end='')
    dxs = df.groupby(level=0).rr_y.diff(periods=1).fillna(df.rr_y)
    des = dxs * df.dedx_y / 1000
    cum_eloss = des.groupby(level=0).cumsum()
    
    data = df[['dedx_y', 'pitch_y']].join(part_df.backtracked_e, on='entry')
    data.backtracked_e -= cum_eloss
    data.rename(columns={'backtracked_e': 'e_y'}, inplace=True)
    tanalyzed = display_uptime(start)
    
    print('Applying Cuts...', end='')
    data = data.groupby(level=0).apply(truncate)
    tcut = display_uptime(tanalyzed)
    
    print('Removing Delta-Rays...', end='')
    data = data.groupby(level=0).apply(delta_rm)
    tdelta_rm = display_uptime(tcut)
    
    data.reset_index(drop=True, inplace=True)
    
    display_uptime(start, 'Done! Total Analysis Time:')
    return data
    

def main():
    start = time.perf_counter()
    full, save = get_inputs()
    tree, savefile = initialize(full, save)
    mask = load_slimming_data(tree)
    data_df, part_df = load_data(tree, mask)
    data = analyze_data(data_df, part_df)
    
    if save:
        print(f"Saving to {savefile}...")
        data.to_csv(savefile, index=False, header=True)
        print("Saved!")
        
    display_uptime(start, "Complete! Total Uptime:")
    
if __name__ == '__main__':
    main()