import numpy as np
import pandas as pd
import uproot
import time
from os.path import realpath, dirname
parent = realpath(dirname(realpath(dirname(realpath(__file__))))) # Don't mind this insanity

# This class is designed to contain a bunch of utility functions which make manipulating and analyzing the data easier
# The methods used in this class are generally not optimized for speed, but good for single-to-few muon analysis
# As such, many functions are defined "muon-wise" - they treat one muon at a time for simplicity.
# Functions within include ability to load muons into RAM, slim muons according to the standard conditions, 
# check whether a muon stops in the detector, generate the eloss for a muon and generate the eloss for all muons.
# I just anticipate spending a while looking at likelihood things and this class will help analyze the raw data to compare
# to the reconstructed data.

class CER():    
    
    def __init__(self, full=False, pitch_lims = (0,70), dedx_max = 100, angle_given = True):
        if full:
            self.treeloc = parent+r"/data/simulated_cosmics_full.root:/nuselection/CalorimetryAnalyzer"
        else:
            self.treeloc = parent+r"/data/simulated_cosmics.root:CalorimetryAnalyzer;9"
        
        self.wire_spacing = 0.3
        self.pitch_min, self.pitch_max = pitch_lims
        if angle_given:
            self.pitch_max = self.wire_spacing / np.cos(pitch_lims[1]*np.pi/180)
            self.pitch_min = self.wire_spacing / np.cos(pitch_lims[0]*np.pi/180)
            
        self.dedx_max = dedx_max
        self.dedx_min = 0
        self.test = ['trk_sce_start_x','trk_sce_start_y','trk_sce_start_z', 
                     'trk_sce_end_x','trk_sce_end_y','trk_sce_end_z',
                     'backtracked_e', 'backtracked_pdg']
        self.anal = ['dedx_y', 'rr_y', 'pitch_y']
        
    def load_muons(self):
        muons = []
        
        with uproot.open(self.treeloc) as tree:
            print("Loading Data...")
            test_muons = tree.arrays(self.test, library='pd')
            anal_muons = tree.arrays(self.anal, library='pd')
            print("Loaded!")
            
            print("Sorting into array of muons...")
            for i in test_muons.index:
                muon = test_muons.iloc[i].squeeze()
                muons.append(muon)

            pidx = anal_muons.index.get_level_values(0).unique()

            for p in pidx:
                anal_muon = anal_muons.loc[p,:]
                for name in self.anal:
                    muons[p][name] = anal_muon[name].squeeze()
            print("Done!")
        # Some loaded muons have no dedx_y for some reason
        # This conditional indexing fixes this
        muons = np.array(muons)[pidx]
        self.muons = muons
    
    def is_non_stopping_muon(self, muon):
        ri = muon.iloc[:3]
        rf = muon.iloc[3:6]
        dimensions = [0, 0, -116, 256, 116, 1036]
        checki = np.zeros(len(dimensions)//2)
        checkf = np.zeros(len(dimensions)//2)

        for i in range(len(dimensions)//2):
            start = dimensions[i]
            end = dimensions[i+3]
            thresh = (end - start)/100
            checki[i] += start+thresh
            checkf[i] += end-thresh

        enters = not ((checki < ri).all() and (checkf > ri).all())
        exits = not ((checki < rf).all() and (checkf > rf).all())
        return enters and exits

    def is_good_muon(self, muon):
        if not (self.dedx_min < muon['backtracked_e'] < self.dedx_max):
            return False
        if np.abs(int(muon['backtracked_pdg'])) != 13:
            return False
        if not (self.pitch_min < muon['pitch_y'][0] < self.pitch_max):
            return False
        if not self.is_non_stopping_muon(muon):
            return False
        return True
    
    def slim_muons(self):
        mask = np.array([ self.is_good_muon(mu) for mu in self.muons ])
        self.muons = self.muons[mask]
        print("Removed", np.sum(~mask), "muons.")
    
    def datapoint_is_invalid(self, de, dedx, lovercostheta, e):
        skip_rest = False
        if de > e:
            skip_rest = True
        if dedx > self.dedx_max:
            skip_rest = True
        if not (self.pitch_min < lovercostheta < self.pitch_max):
            skip_rest = True
        return skip_rest
    
    def generate_eloss(self, muon, verbose=False):
        es = []
        dedxs = []
        msg = ''

        e_losses = muon['dedx_y']
        pitch = muon['pitch_y']
        rr = muon['rr_y']
        e = muon['backtracked_e']

        prev_range = 0

        data_points = rr.index
        for d in data_points:
            x = rr[d]                                    # Particle current x
            dedx = e_losses[d]                           # Particle recent energy loss (MeV/cm)
            lovercostheta = pitch[d]                     # Pitch (For collection wires spaced by 3mm)
            de = (x - prev_range)*dedx/1000              # Approx energy lost since last step (GeV)

            if self.datapoint_is_invalid(de, dedx, lovercostheta, e):
                msg += f'Track becomed invalid at data point {d}'
                break
                
            es.append(e)
            dedxs.append(dedx)
            e -= de                                  # Lower energy accordingly
            prev_range = x                           # Update prev_range
        
        ret = np.array([es, dedxs])
        if verbose:
            ret.append(msg)
        return ret