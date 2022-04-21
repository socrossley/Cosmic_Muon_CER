#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
import sys
import datetime
import gc
import warnings
import time
warnings.filterwarnings('ignore')
import csv
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# In[2]:


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-f", "--full", default=False, type=bool, help="Use the full dataset")
parser.add_argument("-s", "--save", default='', help="Save to file in \'./data\'")
args = vars(parser.parse_args())

# In[3]:


full = args['full']
save = args['save']

data_loc = r"./data/simulated_cosmics.root"
if full:
    data_loc = r"./data/simulated_cosmics_full.root"
    
print("Using data location:", data_loc)

savefile = r'./data/' + save
if save:
    confirm = input(f"Will save output to {savefile}. Confirm (yes/no): ")
    if confirm != 'yes':
        sys.exit(0)
    print('Starting...')


# In[8]:


# Don't import ROOT unless absolutely necessary (takes a long time)
# import ROOT
import uproot


# In[9]:


file = uproot.open(data_loc)
file.values()


# In[5]:


# idx = 3 for full dataset, 0 for smaller dataset
idx = 0
tree = file.values()[idx]
per_particle_variables = ['backtracked_e','backtracked_pdg','backtracked_purity']
variables = ['dedx_y','rr_y','pitch_y']
slimmer_variables = ['trk_sce_start_x','trk_sce_start_y','trk_sce_start_z', 'trk_sce_end_x','trk_sce_end_y','trk_sce_end_z','backtracked_e', 'backtracked_pdg']


# ### Tag for removal
# The following cell populates the list <code>idxs_to_remove</code>, tagging the relevant rows of the dataframe for removal. For now, removal criterion is based solely on whether we think the particle both enters and exits the detector. If it neither enters nor exits at a boundary, particle is tagged for removal.
# - TODO: Speed this up using jit

# In[6]:


def tag_for_removal(slimmerdf):
    # Must match the format of slimmer_variables above, all values in centimeters
    dimensions = [0, -116, 0, 256, 116, 1036]

    idxs_to_remove = []
    pcnt = 0.01
    threshold = [(dimensions[3]-dimensions[0])*pcnt, (dimensions[4]-dimensions[1])*pcnt, (dimensions[5]-dimensions[2])*pcnt]       # Arbitrary threshold; within 3cm of boundary counts as entering or leaving the detector

    for particle in slimmerdf.index:
        start_end_vals, e, pdg = np.split(slimmerdf.iloc[particle].values, [6,7])
        enters = False
        exits = False

        for j in range(len(dimensions)):

            val = start_end_vals[j]
            start = dimensions[j % 3]
            end = dimensions[j % 3 + 3]

            if abs(val - start) < threshold[j % 3]:
                enters = True
                # print("particle", particle, "enters in", j%3)
            elif abs(val - end) < threshold[j % 3]:
                exits = True
                # print("particle", particle, "exits in", j%3)

        # Checks if the particle enters and exits, and also that the particle has a reasonable energy and is a muon
        if not (enters and exits) or e[0] <= 0 or e[0] >= 300 or abs(pdg[0]) != 13:
            idxs_to_remove.append(particle)


    print('Will remove', len(idxs_to_remove), 'particles')
    
    return idxs_to_remove


# ### Generate principal dataframe and slim
# In the future, we will want to do this in batches, as even the slimmed data will be too large to hold in memory all at once. Here, the data is loaded to memory in its entirety, and then slimmed accordingly. Even if we slim better, there is no way around loading the data entirely first before slimming (at least, not that I know of, uproot documentation seems to suggest no - there may be a way in  raw C++)
# - (IMPLEMENTED) It may be better to make two dataframes, one containing the elements that are always the same for a given particle (<code>backtracked_e</code> etc.) and one containing the data points (<code>dedx_y</code>). This should take up less memory as the current implementation of uproot handles these two types of data in the same dataframe by copying the value of backtracked_e for each of the data points in dedx_y, using up a lot more memory than necessary (I think, even if they are just filled with pointers to the same memory address).

# In[7]:


print("Preparing Slimming Mask...")
slimmerdf = tree.arrays(slimmer_variables, library='pd')
idxs_to_remove = tag_for_removal(slimmerdf)


# In[8]:


# Generate DataFrame with the data
# There seems to be a memory leak in pandas https://github.com/pandas-dev/pandas/issues/2659. This casues the 
# allocated memory for the dataframe to be much higher than required. As of now there is no simple fix that I 
# can find, so I will have to work around it.
# Maybe look into this further later if it is a problem with the larger dataset.

print("Generating Principal Dataframe...")
part_df = tree.arrays(per_particle_variables, library='pd')
df = tree.arrays(variables[0], library='pd')
print("Loaded", variables[0], "data...")
size = sys.getsizeof(df)

# True if index is in the indexes tagged for removal
mask = df.index.isin(idxs_to_remove, level=0)

# Slim according to mask
# part_df = part_df.loc[~mask]
df = df.loc[~mask, :]

# This loop loads in the next column of the dataframe, slims it, and appends it to df
for name in variables[1:]:
    next_col = tree.arrays(name, library='pd')
    print("Loaded", name, "data...")
    size += sys.getsizeof(next_col[name])
    next_col = next_col.loc[~mask, :]
    df = df.join(next_col, on=['entry', 'subentry'])

print("Generated!")
print("Original Size:", size/10**6, "MB")
print("Slimmed Size:", sys.getsizeof(df)/10**6, "MB")


# ### Begin Analysis
# The following cell initializes all the necessary variables to be used in the analysis loop.

# #### Initialize some debug counting variables
# These variables keep track of some important data regarding how many particles are ignored, how many data points are ignored, and the number of data points / particles that are ignored for each of the various possible reasons. This way we can keep track of the main reasons why data from certain particles is not being considered.
# - Move info on nege, highe, non-muon, and the number of bad particles to slimming section

# In[9]:


def update_whole_particle_debug_data(e, pur, mu):
    global num_impurities, num_mu, num_amu, num_notmu
    
    # Ideally we don't use this metric for a selection cut
    if pur <= 0.9:
        num_impurities += 1

    if mu == 13:
        num_mu += 1
    elif mu == -13:
        num_amu += 1
    else:
        print('Error: non-muon detected')
        num_notmu += 1


# In[31]:


def datapoint_is_invalid(de, dedx, lovercostheta, e, dedx_cutoff, pitch_high_cutoff):
    global num_highde, num_highdedx, num_highpitch
    skip_rest = False
    
    # TODO: cut drastic changes in pitch
    if de > e:
        skip_rest = True
        num_highde += 1
    if dedx > dedx_cutoff:
        skip_rest = True
        num_highdedx += 1
    if lovercostheta > pitch_high_cutoff:
        skip_rest = True
        num_highpitch += 1
        
    return skip_rest


# In[11]:


def print_debug_data():
    global num_impurities, num_mu, num_amu, num_notmu, num_highde, num_highdedx, num_highpitch, num_particles_partially_skipped, num_skipped_dp
    
    print("Impurities:", num_impurities)
    print("Muons:", num_mu)
    print("Antimuons:", num_amu)
    print("Particles partially skipped:", num_particles_partially_skipped)
    print("Total skipped data points:", num_skipped_dp)
    print("Particles with a high dE data point:", num_highde)
    print("Particles with a high dEdx data point:", num_highdedx)
    print("Particles with a high pitch data point:", num_highpitch)


# In[12]:


def reset_all_counts():
    global num_impurities, num_mu, num_amu, num_notmu, num_highde, num_highdedx, num_highpitch, num_particles_partially_skipped, num_skipped_dp, p_count
    
    num_impurities = 0
    num_mu = 0
    num_amu = 0
    num_particles_partially_skipped = 0
    p_count = 0
    num_skipped_dp = 0
    num_highde = 0
    num_highdedx = 0
    num_highpitch = 0


# ### Principal Analysis Loop
# - TODO: Change how debug counts are handled. Use a dictionary instead and pass that into the relevant functions, rather than the clunky use of global variables.

# In[30]:


#--------------------------------------Initialize debug counting variables-----------------------------------------
# particle counts
num_impurities = 0
num_mu = 0
num_amu = 0
num_particles_partially_skipped = 0
p_count = 0

# data point counts
num_skipped_dp = 0
num_highde = 0
num_highdedx = 0
num_highpitch = 0
    
def analyze_data(df):
    reset_all_counts()
    global p_count, num_particles_partially_skipped, num_skipped_dp
    
    #--------------------------------------Instantiate principal data arrays---------------------------------------
    dedxs = []
    es = []

    #-----------------------------------------Initialize relevant raw data-----------------------------------------
    e_losses_per_step = df['dedx_y']
    true_es = part_df['backtracked_e']
    rrange = df['rr_y']
    pitch = df['pitch_y']
    purity = part_df['backtracked_purity']
    mu_type = part_df['backtracked_pdg']
    particle_idxs = df.index.get_level_values(0).unique()

    #-------------------------------------Initialize selection cut variables---------------------------------------
    dedx_cutoff = 100
    pitch_high_cutoff = 0.3 / np.cos(70*np.pi/180)     # Multiplied by 3mm for wire spacing
    # pitch of greater than 70 degrees wrt collection plane is ignored
    
    #---------------------------------------Initialize analysis time counts----------------------------------------
    tot_particles = len(particle_idxs)
    pcnt_per_count = 100./tot_particles
    count_per_pcnt = 1/pcnt_per_count
    running_count_for_pcnt_increment = 0

    #-----------------------------------------------Start loop-----------------------------------------------------
    start = time.perf_counter()

    print("Analyzing...")
    for p in particle_idxs:

        if p_count > running_count_for_pcnt_increment:
            print(f"{(running_count_for_pcnt_increment / tot_particles)*100:.0f}%        ", end = '\r', flush=True)
            running_count_for_pcnt_increment += count_per_pcnt

        p_count += 1
        data_points = df.loc[p,:].index
        prev_range = 0

        e = true_es[p]                               # True energy of the particle (GeV)
        pur = purity[p]
        mu = mu_type[p]

        update_whole_particle_debug_data(e, pur, mu)        

        for d in data_points:
            i = (p,d)
            x = rrange[i]                                # Particle current x
            dedx = e_losses_per_step[i]                  # Particle recent energy loss (MeV/cm)
            lovercostheta = pitch[i]                         # Pitch (For collection wires spaced by 3mm)
            de = (rrange[i] - prev_range)*dedx/1000      # Approx energy lost since last step (GeV)

            if datapoint_is_invalid(de, dedx, lovercostheta, e, dedx_cutoff, pitch_high_cutoff):
                num_particles_partially_skipped += 1
                num_skipped_dp += len(data_points) - d
                break
            else:
                es.append(e)
                dedxs.append(dedx)
                e -= de                                  # Lower energy accordingly
                prev_range = rrange[i]                   # Update prev_range


    print("100%     ")
    end = time.perf_counter()
    t = end-start
    print(f'Done! Analysis time: {int(t//60)}m {t%60:0.1f}s')  
    print_debug_data()
    
    return es, dedxs


# In[14]:


es, dedxs = analyze_data(df)


# In[2]:


def save_file(path):
    with open(path, 'w', newline='') as save:
        writer = csv.writer(save)
        writer.writerow(es)
        writer.writerow(dedxs)
    print(f"Saved to {path}.")


# In[4]:


if save:
    print(f"Saving to {savefile}...")
    save_file(savefile)