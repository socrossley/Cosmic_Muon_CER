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

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--save", default='', help="Save to file in \'./data\'")
parser.add_argument("-f", "--fitloc", default='stat_fit_data_full.csv', help="Read fit information from \'./data/\'")
parser.add_argument("-c", "--cut", default=False, action='store_true', help="Cut dedx < 1.25 MeV and dedx > 6 MeV")
parser.add_argument("-p", "--pitch-lims", default=[0.3,0.87714132], nargs=2, type=float, help="Limit pitch between pitch-lims in cm (used in conjunction with pitch-limited fitloc)")
parser.add_argument("--full", default=False, action='store_true', help='Load the full dataset (may take a long time)')
args = vars(parser.parse_args())

save = args['save']
fit_data_loc = rf"../data/{args['fitloc']}"
cut = args['cut']
pitch_lims = args['pitch_lims']
full = args['full']

if save:
    print(rf'Will save to: data/{save}')
if full:
    print('Will load full dataset')

fitdata = pd.read_csv(fit_data_loc)

def langau_pdf(dedx, mpv, eta, sig):
    return eta * pylandau.get_langau_pdf(dedx, mpv, eta, sig)

cer = CER(full=full, pitch_lims=pitch_lims, angle_given=False)
cer.load_muons()

def like_max(dedxs):
    if cut:
        dedxs = dedxs[(dedxs > 1.25) & (dedxs < 6)]
    
    landau_params = np.array([ fitdata.iloc[i][:3] for i in range(fitdata.shape[0]) ])
    
    # One big list comprehension for maximum calculation speed
    loglike = np.array([ np.sum([ np.log(langau_pdf(xi, *fj_params)) - np.log(np.sum([ langau_pdf(xi, *fk_params) for fk_params in landau_params])) for xi in dedxs ]) for fj_params in landau_params])
    
    jtilde = np.argmax(loglike)
    e_min_tilde, e_max_tilde = fitdata.iloc[jtilde,-2:]
    return e_min_tilde, e_max_tilde, loglike

def reconstruct_e(muon_idx):  
    es, dedxs = cer.generate_eloss(muon_idx)
    e_min_tilde, e_max_tilde, loglike = like_max(dedxs)
    return e_min_tilde, e_max_tilde, loglike

truth = []
reconstructed = []
loglikes = []
p_count = 0

tot_particles = len(cer.muons.index)
pcnt_per_count = 100./tot_particles
count_per_pcnt = 1/pcnt_per_count
running_count_for_pcnt_increment = 0

print("Generating elosses and reconstructing energy...")
start = time.perf_counter()
for muon_idx in cer.muons.index:
    if p_count > running_count_for_pcnt_increment:
        print(f"{(running_count_for_pcnt_increment / tot_particles)*100:.0f}%   ", end = '\r', flush=True)
        running_count_for_pcnt_increment += count_per_pcnt
        
    p_count += 1
    e_min, e_max, loglike = reconstruct_e(muon_idx)
    
    true_e = cer.muons.backtracked_e.iloc[muon_idx]
    truth.append(true_e)
    
    guess_e = (e_min, e_max)
    reconstructed.append(guess_e)
    loglikes.append(loglike)
    
end = time.perf_counter()
t = end-start
print(f"Done! Analysis time: {int(t//60)}m {t%60:0.1f}s")

like_data_dict = []
for i in range(len(truth)):
    t = truth[i]
    re_min = reconstructed[i][0]
    re_max = reconstructed[i][1]
    
    this_dict = {'truth': t, 'reconstructed_min': re_min, 'reconstructed_max': re_max}
    
    for j in range(len(loglikes[i])):
        like = loglikes[i][j]
        this_dict[f'L{j}'] = like
    
    like_data_dict.append(this_dict)

if save:
    print("Saving likelihood data...")
    like_data = pd.DataFrame.from_dict(like_data_dict)
    like_data.to_csv(rf'../data/{save}', index=False, header=True)
    print('Saved!')