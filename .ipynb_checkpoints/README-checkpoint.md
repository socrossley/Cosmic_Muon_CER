# Cosmic Muon Calorimetric Energy Reconstruction

## Fit data lookup table
| Reconstruction | Key | 
|----------------|-----|
| Base | base | 
| Cut | base_c | 
| Fixed-Width | base_fw |
| Cut and Fixed-Width | base_cfw |
| High-Pitch | hpitch |
| High-Pitch Cut | hpitch_c |
| Low-Pitch | lpitch |
| Low-Pitch Cut | lpitch_c |
| Narrow Energy | narrow | 
| Narrow Energy Cut | narrow_c |

#### Characteristics
 - **Base**: $0.1-100$ GeV muons, $0.3-0.877$ cm pitch range
 - **High-Pitch**: $0.1-100$ GeV muons, $0.7-0.877$ cm pitch range
 - **Low-Pitch**: $0.1-100$ GeV muons, $0.3-0.4$ cm pitch range
 - **Narrow**: $1-10$ GeV muons, $0.3-0.877$ cm pitch range
 - *Cut*: dEdx for reconstruction is excluded unless in range $(1.25, 6)$ MeV/cm (~ $< 0.5%$)
 - *Strict-Cut*: dEdx for reconstruction is excluded unless in range $(1.5, 3.5)$ MeV/cm (~ $< 5%$)
 - *Fixed-Width*: $\sigma, \eta$ in the binned Langaus are fixed to the RMS of all bins.