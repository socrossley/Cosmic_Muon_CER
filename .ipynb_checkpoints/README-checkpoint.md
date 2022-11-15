# Cosmic Muon Calorimetric Energy Reconstruction

The only current way to measure the properties of neutrinos is indirectly via measuring the properties of their decay products. A common neutrino decay product of the Fermilab Booster Neutrino Beam is muons, hence studying their properties is essential to studying neutrinos. 
In this project, I am developing a novel energy reconstruction method for muons in Liquid Argon Time Projection Chambers (LArTPC). This method involves using calorimetry data from muons as they lose energy to the liquid argon medium to determine the initial energy of the muon. My goals are to quantify the energy recolution and bias in muon energy estimation of this approach, and assess the systematic impact of detector resolution and calibration accuracy on calorimetric energy reconstruction.

There are three main components to the analysis: analysis of simulated muon energy loss data at known true muon energies, building of fits to the energy loss distributions of the analyzed data, and reconstruction new muon energy using likelihood maximization on those fits. These components are performed in <code>scripts/Analyze.py</code>, <code>Fit.ipynb</code>, and <code>scripts/recontsruct.py</code>, respectively. Each of these components are complete; the current state of the project is determining confounding variables as to why certain muons have energy loss distributions that deviate significantly from expectation. For muons that truly represent these fitted energy loss distributions, the likelihood was shown to be effective in a MC reconstruction.

This is a research project I am doing under the guidance of Professor David Caratelli at the University of California, Santa Barbara.

The following are personal notes I frequently reference.

## Fit data lookup table
| Reconstruction | Key | 
|----------------|-----|
| Base | base | 
| Cut | base_c |
| Strict Cut | base_sc |
| Fixed-Width | base_fw |
| Cut and Fixed-Width | base_cfw |
| High-Pitch | hpitch |
| High-Pitch Cut | hpitch_c |
| Low-Pitch | lpitch |
| Low-Pitch Cut | lpitch_c |
| Narrow Energy | narrow |
| Narrow Energy Cut | narrow_c |
| Narrow Energy Strict Cut | narrow_sc |
| Narrow Energy Low-Pitch Fixed-Sigma | narrow_lpitch_fs|
| Monte Carlo | mc |

#### Characteristics
 - **Base**: $0.1-100$ GeV muons, $0.3-0.877$ cm pitch range
 - **High-Pitch**: $0.1-100$ GeV muons, $0.7-0.877$ cm pitch range
 - **Low-Pitch**: $0.1-100$ GeV muons, $0.3-0.4$ cm pitch range
 - **Narrow**: $1-10$ GeV muons, $0.3-0.877$ cm pitch range
 - *Cut*: dEdx for reconstruction is excluded unless in range $(1.25, 6)$ MeV/cm (~ $< 0.5%$)
 - *Strict-Cut*: dEdx for reconstruction is excluded unless in range $(1.5, 3.5)$ MeV/cm (~ $< 5%$)
 - *Fixed-Width*: $\sigma, \eta$ in the binned Langaus are fixed to the RMS of all bins.
 - *Fixed-Sigma*: Fit parameters are calculated as normal, then we take the rms of the $\sigma$ values and fix this, then we perform another fit with fixed $\sigma$. 
 - *Monte Carlo*: For now, only one MC run, which uses the narrow, low-pitch, fixed sigma signature.
 
 
## Unbinned Likelihood
$$\ln(\mathcal{L}) = N\ln N + \sum_i \ln\left(\text{LD}\left[\left(\frac{dE}{dx}\right)_i; \text{MPV}(E,p_i), \eta(E,p_i), \sigma(E,p_i)\right]\right)$$
This should be a decent normalized likelihood function for this system, though there are some issues with it outlined below. 

### Current Known Issues
 - Don't know how to account for error in individual $\frac{dE}{dx}$ measurements. Only Poisson error is accounted for.
 - Does not account for the decrease in muon energy along its track. Energy lost can hopfully be added back along each step to get something dependent on only a single $E$ but lost energy should be fairly small.
 - Fairly confidently know the function $\text{MPV}(E,p_i)$, though there is some deviation from the theoretical equation which may come into play.
 - Don't know formulas for $\eta(E,p_i)$, $\sigma(E, p_i)$. Hopefully, there is something in the literature about these parameters, or if they don't vary significantly I can do a linear interpolation of fit data.