# Cosmic Muon Calorimetric Energy Reconstruction

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

#### Characteristics
 - **Base**: $0.1-100$ GeV muons, $0.3-0.877$ cm pitch range
 - **High-Pitch**: $0.1-100$ GeV muons, $0.7-0.877$ cm pitch range
 - **Low-Pitch**: $0.1-100$ GeV muons, $0.3-0.4$ cm pitch range
 - **Narrow**: $1-10$ GeV muons, $0.3-0.877$ cm pitch range
 - *Cut*: dEdx for reconstruction is excluded unless in range $(1.25, 6)$ MeV/cm (~ $< 0.5%$)
 - *Strict-Cut*: dEdx for reconstruction is excluded unless in range $(1.5, 3.5)$ MeV/cm (~ $< 5%$)
 - *Fixed-Width*: $\sigma, \eta$ in the binned Langaus are fixed to the RMS of all bins.
 - *Fixed-Sigma*
 
 
## Unbinned Likelihood
$$\ln(\mathcal{L}) = N\ln N + \sum_i \ln\left(\text{LD}\left[\left(\frac{dE}{dx}\right)_i; \text{MPV}(E,p_i), \eta(E,p_i), \sigma(E,p_i)\right]\right)$$
This should be a decent normalized likelihood function for this system, though there are some issues with it outlined below. 

### Current Known Issues
 - Don't know how to account for error in individual $\frac{dE}{dx}$ measurements. Only Poisson error is accounted for.
 - Does not account for the decrease in muon energy along its track. Energy lost can hopfully be added back along each step to get something dependent on only a single $E$ but lost energy should be fairly small.
 - Fairly confidently know the function $\text{MPV}(E,p_i)$, though there is some deviation from the theoretical equation which may come into play.
 - Don't know formulas for $\eta(E,p_i)$, $\sigma(E, p_i)$. Hopefully, there is something in the literature about these parameters, or if they don't vary significantly I can do a linear interpolation of fit data.