import numpy as np
import pandas as pd
import uproot
import time
from os.path import realpath, dirname
parent = realpath(dirname(realpath(dirname(realpath(__file__))))) # Don't mind this insanity 
from util.theory import langau_pdf

# This class is designed to contain a bunch of utility functions which make manipulating and analyzing the data easier
# The methods used in this class are generally not optimized for speed, but good for single-to-few muon analysis
# As such, many functions are defined "muon-wise" - they treat one muon at a time for simplicity.
# Functions within include ability to load muons into RAM, slim muons according to the standard conditions, 
# check whether a muon stops in the detector, generate the eloss for a muon and generate the eloss for all muons.
# I just anticipate spending a while looking at likelihood things and this class will help analyze the raw data to compare
# to the reconstructed data.

class CER():    
    
    def __init__(self, full=False, pitch_lims=(0,70), dedx_max=100, angle_given=True, distance_thresh=2, e_lims=(0.1,100)):
        if full:
            self.treeloc = parent+r"/data/simulated_cosmics_full.root:/nuselection/CalorimetryAnalyzer"
        else:
            self.treeloc = parent+r"/data/simulated_cosmics.root:CalorimetryAnalyzer;9"
        
        self.distance_thresh = distance_thresh
        self.wire_spacing = 0.3
        self.pitch_min, self.pitch_max = pitch_lims
        self.e_min, self.e_max = e_lims
        if angle_given:
            self.pitch_max = self.wire_spacing / np.cos(pitch_lims[1]*np.pi/180)
            self.pitch_min = self.wire_spacing / np.cos(pitch_lims[0]*np.pi/180)
            
        self.dedx_max = dedx_max
        self.dedx_min = 0
        self.rest_e = 0.105658
        self.test = ['trk_sce_start_x','trk_sce_start_y','trk_sce_start_z', 
                     'trk_sce_end_x','trk_sce_end_y','trk_sce_end_z',
                     'backtracked_e', 'backtracked_pdg']
        self.anal = ['dedx_y', 'rr_y', 'pitch_y']
        
        self._init_dicts()
        
    
    def _init_dicts(self):
        # Lookup dictionaries for the fits, they use the same keys, so the keys can be used universally
        
        fit_data_dir = parent + r'/data/fit_data/'
        self.fits = {'base': fit_data_dir + 'base_fit_data.csv',
                     'base_c': fit_data_dir + 'base_fit_data.csv',
                     'base_sc': fit_data_dir + 'base_fit_data.csv',
                     'base_fw': fit_data_dir + 'fixedwidth_fit_data.csv',
                     'base_cfw': fit_data_dir + 'fixedwidth_fit_data.csv',
                     'hpitch': fit_data_dir + 'highpitch_fit_data.csv',
                     'hpitch_c': fit_data_dir + 'highpitch_fit_data.csv',
                     'lpitch': fit_data_dir + 'lowpitch_fit_data.csv',
                     'lpitch_c': fit_data_dir + 'lowpitch_fit_data.csv',
                     'narrow': fit_data_dir + 'narrow_fit_data.csv',
                     'narrow_c': fit_data_dir + 'narrow_fit_data.csv',
                     'narrow_sc': fit_data_dir + 'narrow_fit_data.csv',
                     'narrow_lpitch_fs': fit_data_dir + 'narrow_lowpitch_fixedsig_fit_data.csv'}
        
        reconstruction_dir = parent + r'/data/reconstructions/'
        self.reconstructions = {'base': reconstruction_dir + 'base_reconstruction.csv',
                                'base_c': reconstruction_dir + 'cut_reconstruction.csv',
                                'base_sc': reconstruction_dir + 'strict_cut_reconstruction.csv',
                                'base_fw': reconstruction_dir + 'fixedwidth_reconstruction.csv',
                                'base_cfw': reconstruction_dir + 'fixedwidth_cut_reconstruction.csv',
                                'hpitch': reconstruction_dir + 'highpitch_reconstruction.csv',
                                'hpitch_c': reconstruction_dir + 'highpitch_cut_reconstruction.csv',
                                'lpitch': reconstruction_dir + 'lowpitch_reconstruction.csv',
                                'lpitch_c': reconstruction_dir + 'lowpitch_cut_reconstruction.csv',
                                'narrow': reconstruction_dir + 'narrow_reconstruction.csv',
                                'narrow_c': reconstruction_dir + 'narrow_cut_reconstruction.csv',
                                'narrow_sc': reconstruction_dir + 'narrow_strict_cut_reconstruction.csv',
                                'narrow_lpitch_fs': reconstruction_dir + 'narrow_lowpitch_fixedsig_reconstruction.csv'}
        
        
    def load_muons(self, slim=True):
        muons = []
        
        with uproot.open(self.treeloc) as tree:
            print("Loading Data...")
            test_muons = tree.arrays(self.test, library='pd')
            anal_muons = tree.arrays(self.anal, library='pd')
            print("Loaded!")
            
            # Some muons in test_muons are not present in anal_muons since pitch is too high
            # this line fixes that
            test_muons = test_muons.loc[anal_muons.index.get_level_values(0).unique()]
            
            if slim:
                test_muons, anal_muons = self.slim_muons(test_muons, anal_muons)
            
        self.muons = Muons(test_muons, anal_muons)
        
    
    def slim_muons(self, test_muons, anal_muons):
        
        def distance_to_edge(r):
            r = np.array(r)
            dimensions = np.array([[0, 256], [-116,116], [0,1036]])
            return  np.min(np.abs(dimensions - r[:, np.newaxis]))
        
        print("Slimming...")
        
        start_dists, end_dists = np.array([ [distance_to_edge(r[:3]), distance_to_edge(r[3:6])] 
                                                     for _, r in test_muons.iterrows() ]).T

        is_muon = np.abs(test_muons.backtracked_pdg) == 13
        has_bethe_energy = ((self.e_min < test_muons.backtracked_e) &
                            (test_muons.backtracked_e < self.e_max))
        has_good_pitch = ((self.pitch_min < anal_muons.pitch_y.loc[:,0]) & 
                          (anal_muons.pitch_y.loc[:,0] < self.pitch_max))
        is_non_stopping = ((start_dists < self.distance_thresh) & 
                           (end_dists < self.distance_thresh))

        mask = (is_muon & has_bethe_energy & has_good_pitch & is_non_stopping).to_numpy()
        
        # Broadcast mask to multiindex shape
        _, num_dp_per_muon = np.unique(anal_muons.index.get_level_values(0), return_counts=True)
        multi_mask = np.repeat(mask, num_dp_per_muon)
        print("Will remove", np.sum(~mask), "particles")

        return test_muons.loc[mask], anal_muons.loc[multi_mask, :]
    
    
    def datapoint_is_invalid(self, de, dedx, lovercostheta, e):
        skip_rest = False
        if de > e:
            skip_rest = True
        if dedx > self.dedx_max:
            skip_rest = True
        if not (self.pitch_min < lovercostheta < self.pitch_max):
            skip_rest = True
        return skip_rest
    
    
    def generate_eloss(self, muon_idx, verbose=False):
        es = []
        dedxs = []
        msg = ''

        e_losses = self.muons.dedx_y.loc[muon_idx]
        pitch = self.muons.pitch_y.loc[muon_idx]
        rr = self.muons.rr_y.loc[muon_idx]
        e = self.muons.backtracked_e.loc[muon_idx]
        
        prev_range = 0
        
        data_points = rr.index
        for d in data_points:
            x = rr[d]                                    # Particle current x
            dedx = e_losses[d]                           # Particle recent energy loss (MeV/cm)
            lovercostheta = pitch[d]                     # Pitch (For collection wires spaced by 3mm)
            de = (x - prev_range)*dedx/1000              # Approx energy lost since last step (GeV)

            if self.datapoint_is_invalid(de, dedx, lovercostheta, e):
                msg += f'Track becomed invalid at data point {d}\n'
                break
                
            es.append(e)
            dedxs.append(dedx)
            e -= de                                  # Lower energy accordingly
            prev_range = x                           # Update prev_range
        
        ret = np.array([es, dedxs])
        if verbose:
            ret = np.append(ret, [msg])
        return ret
    
    '''
    # Precompiled Eloss todo and test
    def generate_eloss(muon_idx):
        e_losses = self.muons.dedx_y.loc[muon_idx]
        pitch = self.muons.pitch_y.loc[muon_idx]
        rr = self.muons.rr_y.loc[muon_idx]
        e = self.muons.backtracked_e.loc[muon_idx]
    '''
    
    def binned_like(self, dedxs, fit_key, cum=False):
        # If this is a cut run, attenuate the dedxs
        flag = fit_key.split('_')[-1]
        if 'sc' in flag:
            dedxs = dedxs[(dedxs > 1.5) & (dedxs < 3.5)]
        elif 'c' in flag:
            dedxs = dedxs[(dedxs > 1.25) & (dedxs < 6)]
            
        summate = lambda a: a
        if cum:
            summate = np.cumsum
        
        fitdata = pd.read_csv(self.fits[fit_key])
        landau_params = np.array([ fitdata.iloc[i][:3] for i in range(fitdata.shape[0]) ])
        
        loglike = np.array([ summate([ np.log(langau_pdf(xi, *fj_params)) - np.log(np.sum([ langau_pdf(xi, *fk_params) for fk_params in landau_params])) for xi in dedxs ]) for fj_params in landau_params ])
        
        return loglike, dedxs
    
    # Returns a cumulative array of likelihood
    def cum_binned_like(self, dedxs, fitkey):
        return self.binned_like(dedxs, fitkey, cum=True)
    
    
class Muons():
    
    def __init__(self, test_muons, anal_muons):
        self.index = np.arange(test_muons.shape[0])
        self.test_cols = test_muons.columns
        self.anal_cols = anal_muons.columns
        
        self.test_muons = test_muons.set_index(self.index)
        idx = anal_muons.index.remove_unused_levels()
        self.anal_muons = anal_muons.set_index(idx.set_levels(np.arange(test_muons.shape[0]), level=0))
        
        
    def __getattr__(self, name):
        if name in self.test_cols:
            return self.test_muons[name]
        
        if name in self.anal_cols:
            return self.anal_muons[name]
        
        raise AttributeError()
