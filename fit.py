import numpy as np
import uproot
import matplotlib.pyplot as plt
import matplotlib
from iminuit import Minuit
import iminuit
from scipy.optimize import curve_fit

tree = uproot.open("~/EPPT/MPAGS-EPPT-B5/B5.root")['B5']

detector1_x = tree['Dc1HitsVector_x'].array()
detector2_x = tree['Dc2HitsVector_x'].array()
detector1_y = tree['Dc1HitsVector_y'].array()
detector2_y = tree['Dc2HitsVector_y'].array()
detector1_z = tree['Dc1HitsVector_z'].array()
detector2_z = tree['Dc2HitsVector_z'].array()

zhits =  np.array([0, 0.5, 1, 1.5, 2, 8.5, 9, 9.5, 10, 10.5])
x_hits = []
for i in range(len(detector1_x)):
    hits = []
    if len(detector1_x[i])==5 and len(detector2_x[i])==5 and len(detector1_z[i])==5 and len(detector2_z[i]) == 5:
        hits.append(detector1_x[i])
        hits.append(detector2_x[i])
        x_hits.append(np.concatenate(hits))
x_hits = [x*1E-3 for x in x_hits]
def x_position(z_arr, p, m, c):
        x_pos = []
        B = 1
        radius = p/(0.3*B)
        perp_grad = -1/m
        if np.isfinite(perp_grad) == False:
            perp_grad = 1
        if np.isfinite(m) == False:
            m = 1E-6
        theta = (np.arctan2(1, perp_grad))
        #theta = np.abs(np.arctan(m))
        delta_z = radius*np.sin(theta)
        delta_x = -radius*np.cos(theta)
        z = 5.25
        x = m*5.25 + c
        if m > 0:
                centre = np.array([z+delta_z, x+delta_x])
        if m < 0:
                centre = np.array([z-delta_z, x+delta_x])
        z_exit = 7.25
        x_exit = centre[1] + np.sqrt(np.abs(radius**2 - (7.25 - centre[0])**2))
        exit_gradient = -1/((x_exit - centre[1])/(z_exit-centre[0]))
        if np.isfinite(exit_gradient) == False:
                exit_gradient=1
        exit_intercept = x_exit - exit_gradient
        for z in z_arr:
            if z <= 5.25:
                x_pos.append(z*m + c)
            elif 5.25<z and z<7.25:
                x_pos.append(centre[1] + np.sqrt(np.abs(radius**2 - (z - centre[0])**2)))
            elif z>=7.25:
                x_pos.append(exit_gradient*(z-6.25) + exit_intercept)
        return np.array(x_pos)


def chi_2(p, m , c):

    return np.sum((((x_hits - x_position(zhits, p, m, c)))/(100E-6))**2)



m = Minuit(chi_2, p=100, m=1E-5, c=1E-1)
m.migrad()
m.hesse()

print(m.values['p'], m.values['m'], m.values['c'])

"""
x_fit = np.linspace(0, 11, 100)
fit = np.array([x_position(x_fit, m.values['p'], m.values['m'], m.values['c'])])
fig = plt.figure(figsize=(7,7))
c = plt.get_cmap('turbo', 8)
plt.scatter(zhits, x_hits, color=c(2), label='data' )
plt.scatter(zhits, x_position(zhits, m.values['p'], m.values['m'], m.values['c']), color=c(6), label='Fit')
plt.plot(x_fit, x_position(x_fit, m.values['p'], m.values['m'], m.values['c']), color=c(6), label='fit')
plt.vlines(5.25, -0.01, 0.01)
plt.vlines(7.25, -0.01,0.01)
plt.xlabel('z position')
plt.ylabel('x position')
plt.legend()
plt.show()
"""
pt = []

for x in x_hits:
        def chi_2(p, m , c):

                return np.sum((((x - x_position(zhits, p, m, c)))/(100E-6))**2)
        m = Minuit(chi_2,p=100, m=1E-5, c=1E-1)
        m.migrad()
        m.hesse()
        pt.append(m.values['p'])

def gaus(x, c, mu, sig):
    return c*np.exp(-0.5*((x-mu)/sig)**2)

c = plt.get_cmap('turbo', 1)
plt.figure(figsize=(6,6))
plt.hist(pt, bins=np.linspace(90,110, 51), histtype='stepfilled')
binning = np.linspace(100.94, 101.74 , 51)
h_pt, bins = np.histogram(pt, bins=binning)
p0 = [5500, 101, 0.2]
popt, pcov = curve_fit(gaus, bins[:-1], h_pt, p0=p0)
y = gaus(binning, *popt)
plt.plot(binning, 25*y, label=f'Gaussian fit, \n mean = {popt[1]:.2f}, \n  sigma = {popt[2]:.1f}')
plt.xlabel('Transverse Momentum [GeV]')
plt.ylabel('Counts')
plt.legend(frameon=False)
plt.show()
                
