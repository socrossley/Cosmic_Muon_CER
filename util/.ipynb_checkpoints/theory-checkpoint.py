import numpy as np

Z = 18
A = 39.948             # g / mol
I = 188.0 * (10**(-6)) # MeV
K = 0.307              # MeV * cm^2 / mol
Mmu = 105.658          # MeV for muon
Me  = 0.51             # MeV for electron
rho = 1.396            # g/cm3

def beta(gamma):
    return np.sqrt(1-(1./(gamma**2)))

def gamma(KE,mass):
    return (KE/mass)+1

def Wmax (KE,mass):
    g = gamma(KE,mass)
    b = beta(g)
    num = 2*Me*((b*g)**2)
    den = 1 + 2*g*Me/mass + (Me/mass)**2
    return num/den

# density correction for LAr
def density(bg):

    # constants and variable names obtained from :
    # PDG elos muons table [http://pdg.lbl.gov/2016/AtomicNuclearProperties/MUE/muE_liquid_argon.pdf]
    
    C  = -5.2146
    X0 = 0.2
    X1 = 3.0
    a  = 0.19559
    m  = 3.0
    N    = 2 * np.log(10)
    
    x = np.log10(bg)

    def smallx(x):
        return 0
    def midx(x):
        addition = a * ((X1-x)**m)
        return N * x + C + addition
    def largex(x):
        return N * x + C
    
    condlist = [x < X0, (x >= X0) & (x < X1), x >= X1]
    funclist = [smallx, midx, largex]
    
    return np.piecewise(x, condlist, funclist)
    
    
def dpdx(KE,x,mass,I=I):
    g = gamma(KE,mass)
    b = beta(g)
    epsilon = (K/2.)*(Z/A)*(x*rho/(b*b))
    A0 = (2*Me*(b*g)**2)/I
    A1 = epsilon/I
    return (1/x) * epsilon * (np.log(A0) + np.log(A1) + 0.2 - (b*b) - density(b*g))

# KE in MeV
# x in cm
# mass in MeV
# in MeV/cm
def dedx(KE,mass,I=I,dens=True):
    g = gamma(KE,mass)
    b = beta(g)
    F = K * (Z/A)*(1/b)**2 * rho
    wmax = Wmax(KE,mass)
    a0 = 0.5*np.log( 2*Me*(b*g)**2 * wmax / (I*I) )
    ret = a0 - b*b
    if (dens == True):
        ret -= density(b*g)/2.
    return F * ret

def dedx_R(KE, mass, wcut, K=K, I=I, dens=True):
    g = gamma(KE,mass)
    b = beta(g)
    F = K * (Z/A)*(1/b)**2 * rho
    wmax = Wmax(KE, mass)
    a0 = 0.5*np.log( 2*Me*(b*g)**2 * wcut / (I*I) )
    ret = a0 - b*b / 2 * (1 + wcut/wmax)
    if dens == True:
        ret -= density(b*g)/2
    return F * ret