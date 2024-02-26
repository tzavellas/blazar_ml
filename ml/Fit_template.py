from astropy import constants as const
import corner
import emcee
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tensorflow import keras
from common import log_gamma_range
import time
from multiprocessing import Pool


c = const.c.cgs.value
me = const.m_e.cgs.value
h = const.h.cgs.value
sigma_T = const.sigma_T.cgs.value
D = 3262 * 3.1e24  # cm
# q = const.e.esu.value
model_path = '/DATA/hea_ml_generated_files/tf2_models/hea_gru.h5'
# model_path = '/DATA/hea_ml_generated_files/tf2_models/hea_lstm.h5'
tf_model = keras.models.load_model(model_path)


def de_normalize(data, min_val=-30, max_val=0):
    return min_val + (max_val - min_val) * data


def Model(x, P1, P2, P3, P4, P5, P6, P7):
    x_grid = np.linspace(-15, 10, 500)
    params = np.expand_dims(np.array([P1, P2, P3, P4, P5, P6]), axis=0)
    y_pred = tf_model.predict(params, verbose=0)
    y_pred_d = de_normalize(y_pred)

    xp = x_grid + np.log10(me * c**2 / h) + P7
    flux_norm = P1 + np.log10(4 * np.pi * me * c **
                              3 / sigma_T / 3) - np.log10(4 * np.pi * D**2)

    boost = 4 * P7

    return np.interp(x, xp, y_pred_d[0] + flux_norm + boost)


def log_likelihood(theta, x, y, yerr):
    P1, P2, P3, P4, P5, P6, P7, log_f = theta
    model = Model(x, P1, P2, P3, P4, P5, P6, P7)
#     sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    sigma2 = yerr**2 + np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta):
    P1, P2, P3, P4, P5, P6, P7, log_f = theta
    P4_min, P4_max = log_gamma_range(P3, P1, P2)
    if (14 < P1 < 17) and (-2 < P2 < 2) and (0.1 < P3 < 4) and\
        (P4_min < P4 < P4_max) and (-5 < P5 < -1) and (1.5 < P6 < 3) and\
            (0 < P7 < 3) and (-10.0 < log_f < 1.0):
        return 0.0
    return -np.inf


def log_probability(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

data_fermi = np.loadtxt('/home/tzavellas/hea_ml_generated_files/fermi_data.txt', usecols=range(0, 4))
dat_fermi = pd.DataFrame(data_fermi, columns=['E0', 'vFv0', 'vFve', 'flag'])

dat_fermi['E'] = np.log10(dat_fermi['E0'])
dat_fermi['vFv'] = np.log10(dat_fermi['vFv0']*1.6e-6)

mask = (dat_fermi['flag'] == 0)

data = np.loadtxt('OUsed_5BZBJ09553551-11jan2020.txt', usecols=range(0, 4))
dat = pd.DataFrame(data, columns=['E', 'eB', 'vFv', 'vFve'])

x = np.linspace(5, 30, 100)
params = [16.25808, 1.538818, 1.980959, 6.581445, -3.840278, 2.438278, 1]
params = [16.25808, 1.538818, 3.9, 6.581445, -3.840278, 2, 1]


y_model = Model(x, *params)

plt.figure(figsize=(10, 6), dpi=80)
# plt.plot(fx/secd, fy, 'm')
plt.plot(dat['E'], dat['vFv'], 'ob')
plt.plot(dat_fermi['E'][mask], dat_fermi['vFv'][mask], 'om')
plt.errorbar(dat['E'], dat['vFv'],yerr=dat['vFve'],marker='.',linestyle='None')
plt.plot(x, y_model, 'xr')
plt.show()


down_lims = [14, -2, 0.1, 1, -5, 1.5, 0]
up_lims = [17, 2, 4, 8, -1, 3 , 3]
guess_par = params

xd = dat['E'] 
yd = dat['vFv']
yderr = dat['vFve']

xd = xd.append(dat_fermi['E'],ignore_index=True)
yd = yd.append(dat_fermi['vFv'],ignore_index=True)
yderr = yderr.append(0.2+0*dat_fermi['vFv'],ignore_index=True)
# fit_params, pcov = curve_fit(Model, xd, yd,sigma=yderr,bounds=(down_lims,up_lims),p0=guess_par)
# print(fit_params)


init = [*params,-5]

pos = init + 1e-2 * np.random.randn(32, len(init))
nwalkers, ndim = pos.shape

start_time = time.time()

print('start sampler')

# with Pool() as pool:
#     sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xd, yd, yderr), pool = pool)
#     sampler.run_mcmc(pos, 10, progress=True)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(xd, yd, yderr))
sampler.run_mcmc(pos, 10000, progress=True)


print("--- %s seconds ---" % (time.time() - start_time))

fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
samples = sampler.get_chain()
labels = ["P1", "P2", 'P3', 'P4', 'P5', 'P6', 'P7','logf']
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
plt.show()

# axes[-1].set_xlabel("step number");


# # tau = sampler.get_autocorr_time()
# # print(tau)

flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)
print(flat_samples.shape)


# for i in range(ndim):
#     mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
#     q = np.diff(mcmc)
#     txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
#     txt = txt.format(mcmc[1], q[0], q[1], labels[i])
    # display(Math(txt))


fig = corner.corner(
    flat_samples, labels=labels
);

##Plots chain with model ---------------------------------------
inds = np.random.randint(len(flat_samples), size=20)
plt.plot(dat['E'], dat['vFv'], 'ob')
plt.plot(dat_fermi['E'][mask], dat_fermi['vFv'][mask], 'om')
plt.errorbar(dat['E'], dat['vFv'],yerr=dat['vFve'],marker='.',linestyle='None')
plt.ylim([-13,-10])
for ind in inds:
    sample = flat_samples[ind]
    y_model = Model(x, *sample[:-1])
    plt.plot(x, y_model, ':r')
plt.show()
    