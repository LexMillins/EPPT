import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

def gaus(x, c, mu, sig):
    return c*np.exp(-0.5*((x-mu)/sig)**2)
    

tree = uproot.open("~/EPPT/MPAGS-EPPT-B5/B5.root")['B5']

detector1_x = tree['Dc1HitsVector_x'].array()
detector2_x = tree['Dc2HitsVector_x'].array()
detector1_y = tree['Dc1HitsVector_y'].array()
detector2_y = tree['Dc2HitsVector_y'].array()
detector1_z = tree['Dc1HitsVector_z'].array()
detector2_z = tree['Dc2HitsVector_z'].array()

num=0
a1_arr = []
a2_arr = []
b1_arr = []
b2_arr = []
a1_errors = []
a2_errors = []

c = plt.get_cmap('turbo', 2*len(detector1_x))
plt.figure(figsize=(10,10))
print(len(detector1_z))
for xhit, zhit in zip(detector1_x, detector1_z):
    num +=1
    coeff, cov =  np.polyfit(xhit, zhit*1000, 1, cov=True)
    a1 = coeff[0]
    b1 = coeff[1]
    x1 = np.linspace(-1, 1, 101)
    y = [a1*x + b1 for x in x1]
    yerr = np.sqrt(np.diag(cov))
    print(cov)
    if b1 < 0:
        a1_arr.append(a1)
        b1_arr.append(b1)
        a1_errors.append(yerr)

num = 0
for xhit, zhit in zip(detector2_x, detector2_z):
    #zhit = zhit + 4.5
    num +=1
    coeff2, cov2 = np.polyfit(xhit, zhit*1000, 1, cov=True)
    a2 = coeff2[0]
    b2 = coeff2[1]
    x2 = np.linspace(-15, -5, 101)
    y2 = [a2*x + b2 for x in x2]
    y2err = np.sqrt(np.diag(cov2))
    if b2 < 0:
        a2_arr.append(a2)
        b2_arr.append(b2)
        a2_errors.append(a2)

angles = []
angle_errors = []
moms = []
mom_errors = []
for i in range(len(a1_arr)):
    tan_angle = (a1_arr[i] - a2_arr[i])/(1+a1_arr[i]*a2_arr[i])
    angle = np.arctan(tan_angle)
    sigma_theta = np.sqrt(1/(a1_arr[i]**2+1)*a1_errors[i]**2 + 1/(a2_arr[i]**2+1)*a2_errors[i]**2)
    angles.append(angle)
    angle_errors.append(sigma_theta)
    mom = np.abs((0.3*0.5)/angle)
    sigma_mom = (mom/angle)*sigma_theta
    moms.append(mom)
    mom_errors.append(np.abs(sigma_mom))

h_mom, bins = np.histogram(moms, bins=np.linspace(97.43, 102.8, 101))
p0 = [100, 100, 5]
popt, pcov = curve_fit(gaus, bins[:-1], h_mom, p0=p0)
y = gaus(bins, *popt)

plt.hist(np.concatenate(mom_errors))
plt.show()
"""
plt.style.use("seaborn-talk")
#### Manual global changes to style
plt.rc('legend', fontsize = 10)
plt.rc('legend', title_fontsize = 10)
plt.rc('font', size = 10)
plt.rc('figure', figsize = (7,7), dpi = 100)
global_dpi = 100
matplotlib.rcParams['figure.dpi'] = global_dpi
c = plt.get_cmap('jet', 5)
plt.figure(figsize=(7,7))
plt.hist(moms, bins=np.linspace(80, 120, 101), color=c(1), histtype='stepfilled', edgecolor='k', density=True)
plt.plot(bins, 0.01*y, label=f'Mean = {popt[1]:.2f}, sigma = {popt[2]:.2f}')
plt.ylabel('Density')
plt.xlabel('Momentum [GeV]')
plt.legend(frameon=False)
plt.show()
"""





