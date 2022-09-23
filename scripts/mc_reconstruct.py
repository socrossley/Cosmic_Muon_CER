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
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import warnings
warnings.filterwarnings('ignore')

# Parse argments inputted on the command line
parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("-s", "--save", default='', help="Save to file in \'./data/reconstructions\'")
parser.add_argument("-n", "--num-per-ebin", default='200', type=int, help="How many MC muons to reconstruct per energy bin")
parser.add_argument("--mc-only", default=False, action='store_true', help="Produce the MC muon dedxs only (no reconstruction)")
args = vars(parser.parse_args())

save = args['save']
muons_per_ebin = args['num_per_ebin']
mc_only = args['mc_only']

fitdf = pd.read_csv('../data/fit_data/narrow_lowpitch_fixedsig_fit_data.csv')
langau_params = fitdf[['mpv', 'eta', 'sigma']]
rng = np.random.default_rng()

                        
def sample_from_langau(mpv, eta, sigma):
    prob = rng.uniform()
    to_solve = lambda b: quad(theory.langau_pdf, -np.inf, b, args=(mpv, eta, sigma))[0]-prob
    dedx = fsolve(to_solve, 2)[0]
    return prob, dedx
                   
                        
def lognorm(x, s, loc, scale):
    return scipy.stats.lognorm.pdf(x, s, loc, scale)
                        

# Hardcoded track lenght distribution parameters
s, loc, scale = ( 2.01980595e-01, -1.23848172e+03, 1.66933758e+03 )
def rand_trkl():
    val = int(scipy.stats.lognorm.rvs(s, loc, scale))
    while val < 2:
        val = int(scipy.stats.lognorm.rvs(s, loc, scale))
    return val

# Perform the MC generation
print("Generating MC muon tracks...")
start = time.perf_counter()
dedxs_dict = {} 
for i, params in langau_params.iterrows():
    print('Bin '+ str(i) + '...')
    dedxs_per_ebin = []
    for j in range(muons_per_ebin):
        trkl = rand_trkl()
        
        dedxs = []
        for k in range(trkl):
            prob, dedx = sample_from_langau(*params)
            dedxs.append(dedx)
        
        dedxs_per_ebin.append(dedxs)
    dedxs_dict[str(i)] = dedxs_per_ebin
    
end = time.perf_counter()
t = end-start
print('Generated!')
print(f'Total time {int(t//60):d}m {t%60:.1f}s')

if mc_only:
    df = pd.DataFrame(dedxs_dict)
    df.to_csv('../data/mc_dedxs.csv')
    print("dedx data saved!")
    sys.exit(0)

# Same likelihood as used before
def like_max(dedxs):
    landau_params = np.array([ langau_params.iloc[i] for i in range(fitdf.shape[0]) ])
    
    # One big list comprehension for maximum calculation speed
    loglike = np.array([ np.sum([ np.log(theory.langau_pdf(xi, *fj_params)) - np.log(np.sum([ theory.langau_pdf(xi, *fk_params) for fk_params in landau_params])) for xi in dedxs ]) for fj_params in landau_params])
    
    jtilde = np.argmax(loglike)
    e_min_tilde, e_max_tilde = fitdf[['e_min', 'e_max']].iloc[jtilde]
    return e_min_tilde, e_max_tilde, loglike


# Same script as in reconstruct.py modified for the MC data
truth = []
reconstructed = []
loglikes = []
p_count = 0

tot_particles = muons_per_ebin * fitdf.shape[0]
pcnt_per_count = 100./tot_particles
count_per_pcnt = 1/pcnt_per_count
running_count_for_pcnt_increment = 0

print("Reconstructing energies...")
start = time.perf_counter()
for key, value in dedxs_dict.items():
    
    for muon_dedxs in value:
        
        if p_count > running_count_for_pcnt_increment:
            print(f"{(running_count_for_pcnt_increment / tot_particles)*100:.0f}%   ", end = '\r', flush=True)
            running_count_for_pcnt_increment += count_per_pcnt

        p_count += 1
        e_min, e_max, loglike = like_max(muon_dedxs)

        true_e = fitdf[['e_min', 'e_max']].iloc[int(key)].mean()
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
    like_data.to_csv(rf'../data/reconstructions/{save}', index=False, header=True)
    print('Saved!')