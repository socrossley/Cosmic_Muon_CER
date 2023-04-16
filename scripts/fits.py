import sys
from os.path import realpath, dirname
sys.path.insert(0, dirname(realpath('')))

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
import numpy as np
import util.theory as theory

# So far, just generates theoretical fit data

def get_inputs():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-s", "--save", default=False, action='store_true', help="Save to file in \'./data/fit_data/th\'")
    parser.add_argument("-n", "--num-bins", default=30, type=int, help="Number of bins within 1-10 GeV range. More bins are added to the ends")
    parser.add_argument("-e", "--error", default=0, type=float, help="Error percentage to introduce to the generated Langau fits")
    parser.add_argument("--energy-range", default=[1,10], nargs=2, type=float, help="Energy range over which to generate Langau fits.")
    
    args = vars(parser.parse_args())
    return args


def get_theoretical_fit_data(num_bins, min_e=1, max_e=10, spread_pct=0):
    thickness = 0.35   
    
    bin_width = (max_e-min_e) / num_bins
    
    # Add some bins for padding
    left_bin_edge_padding = 0
    right_bin_edge_padding = 0
    left_edge = min_e - left_bin_edge_padding*bin_width
    right_edge = max_e + (right_bin_edge_padding+0.5)*bin_width
    num_bins += left_bin_edge_padding + right_bin_edge_padding
    
    if left_edge < 0.2:
        raise ValueError('Increase number of bins!')
    
    bin_edges = np.arange(left_edge, right_edge, bin_width)
    e_max = bin_edges[1:]
    e_min = bin_edges[:-1]
    
    TE_GeV = (e_min + e_max)/2
    KE_MeV = TE_GeV * 1000 - theory.Mmu
    
    mpv = theory.dpdx(KE_MeV, thickness, theory.Mmu)
    sigma = mpv*spread_pct/100
    eta = np.array([0.076] * num_bins)
    
    th_fitdf = pd.DataFrame(np.transpose([mpv, eta, sigma, e_min, e_max]), 
                            columns=['mpv', 'eta', 'sigma', 'e_min', 'e_max'])
    return th_fitdf


def main():
    args = get_inputs()
    df = get_theoretical_fit_data(args['num_bins'], min_e=args['energy_range'][0], max_e=args['energy_range'][1], spread_pct = args['error'])
    if args['save']:
        savefile = r'../data/fit_data/th/' + str(args['error']) + 'ps.csv'
        print(f'Saving to {savefile}')
        df.to_csv(savefile, index=False, header=True)
    print('Saved!')
        
if __name__ == '__main__':
    main()