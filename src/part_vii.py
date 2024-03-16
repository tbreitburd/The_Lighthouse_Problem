import numpy as np
import zeus
import os
import funcs
from emcee.autocorr import integrated_time

# Ignore seaborn FutureWarnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


print("---------------------------------")
print("----------- Part VII --------------")
print("---------------------------------")

# -------------------------------------------------------------
# -------------------------------------------------------------
# Get the data from the txt file
# -------------------------------------------------------------
# -------------------------------------------------------------

path = "lighthouse_flash_data.txt"

# Get the parent directory of the current script
proj_dir = os.getcwd()
data_path = os.path.join(proj_dir, path)

# Read the data
data = np.loadtxt(data_path)
flash_locs = data[:, 0]
log_flash_intensities = np.log(data[:, 1])

print("---------------------------------")
print("---------------------------------")
print("(a) The Data")
print("The observed flashes have locations and intensities:")
print(data)

# Set the limits for alpha and beta
alpha_lim = 100
beta_lim = 100
intensity_min = 0.0001
intensity_max = 50

# Plot the data as a diagram of the lighthouse and the flashes
# for a hypothetical lighthouse at position alpha = 0 and beta = 0.1
funcs.plot_lighthouse(flash_locs, (0, 0.1))

# Visualise the Cauchy distribution for the lighthouse problem,
# again for hypothetical alpha = 0 and beta = 0.1
x = np.linspace(-10, 10, 1000)
funcs.plot_lighthouse_cauchy(0, 2, x)

# --------------------------------------------------------------
# --------------------------------------------------------------
# Sample from the posterior
# --------------------------------------------------------------
# --------------------------------------------------------------
print("---------------------------------")
print("---------------------------------")
print("(b) Sampling from the posterior")
print("        ")

# Set the random seed
np.random.seed(42)
# ---------------------------------
# Using Metropolis-Hastings
# ---------------------------------
print("------ Metropolis-Hastings ------")
print(
    "Using the Metropolis-Hastings algorithm,"
    + " with a multivariate normal proposal distribution:"
)


nsteps, ndim = 500000, 3  # Define the number of steps and the number of dimensions
init = [
    np.random.uniform(-alpha_lim, alpha_lim),
    np.random.uniform(0, beta_lim),
    np.random.uniform(intensity_min, intensity_max),
]  # Starting point of the chain
cov = np.array([[1.5, 0.0, 0.0], [0.0, 1.5, 0.0], [0.0, 0.0, 1.5]])

# Run the Metropolis-Hastings algorithm
chain_MH, accept_frac_MH = funcs.Metropolis_Hastings(
    nsteps, ndim, funcs.log_posterior_vii, cov, init
)

print("The acceptance fraction is for the Metropolis Hastings algorithm is:")
print(accept_frac_MH)

# Plot the chain to see the burn-in
funcs.plot_chain(chain_MH, 1000, ndim, "MH")

burn_in = 1000

# Remove burn-in
chain_MH = chain_MH[burn_in:, :]

# Get the autocorrelation time, tau, for the chain
taus_MH = (
    integrated_time(chain_MH[:, 0]),
    integrated_time(chain_MH[:, 1]),
    integrated_time(chain_MH[:, 2]),
)
print("Autocorrelation:", taus_MH[0], taus_MH[1], taus_MH[2])

tau_MH = max(taus_MH)
print("tau_MH = ", tau_MH)


# Gelman-Rubin statistic
# Sample multiple chains of smaller size using MH
# and calculate the Gelman-Rubin statistic using a rolling window
nchains = 6
nsteps = 10000

# Run the Metropolis-Hastings algorithm
chains_MH = np.zeros((nsteps, nchains, ndim))
for i in range(nchains):
    init = [
        np.random.uniform(-alpha_lim, alpha_lim),
        np.random.uniform(0, beta_lim),
        np.random.uniform(intensity_min, intensity_max),
    ]
    chains_MH[:, i, :], _ = funcs.Metropolis_Hastings(
        nsteps, ndim, funcs.log_posterior_vii, cov, init
    )

# Calculate the Gelman-Rubin statistic
R_MH = funcs.gelman_rubin(chains_MH, 100, 10, 3)

# Plot the Gelman-Rubin statistic
funcs.plot_gelman_rubin(R_MH, ndim, "MH")

print(tau_MH)
print(chain_MH.shape)
# Thin the chain
chain_MH = chain_MH[:: 2 * int(tau_MH[0]), :]
print(chain_MH.shape)

# Plot the distribution of the samples in a corner plot
funcs.plot_corner(chain_MH, ndim, "MH")

# Get estimates of alpha and beta
alpha_est_MH, beta_est_MH, Intensity_0_est_MH = np.mean(chain_MH, axis=0)
alpha_sig_MH, beta_sig_MH, Intensity_0_sig_MH = np.std(chain_MH, axis=0)

print("The estimated alpha, beta, and intensity are:")
print("alpha_MH = ", alpha_est_MH, " ± ", alpha_sig_MH)
print("beta_MH = ", beta_est_MH, " ± ", beta_sig_MH)
print("Intensity_0_MH = ", Intensity_0_est_MH, " ± ", Intensity_0_sig_MH)


# ---------------------------------
# Using the zeus sampler (ensemble sampler)
# ---------------------------------
print("        ")
print("------ Zeus Slice Sampling ------")
print("Using the zeus sampler, with slice sampling:")

np.random.seed(42)

nsteps, nwalkers, ndim = 10000, 10, 3
start = np.array(
    [
        np.random.uniform(-alpha_lim, alpha_lim, 10),
        np.random.uniform(0, beta_lim, 10),
        np.random.uniform(intensity_min, intensity_max, 10),
    ]
).T
# start = np.abs(np.random.randn(nwalkers, ndim))

sampler_SS = zeus.EnsembleSampler(nwalkers, ndim, funcs.log_posterior_vii)

sampler_SS.run_mcmc(start, nsteps)
chain_SS = sampler_SS.get_chain()


# Calculate the Gelman-Rubin statistic
R_SS = funcs.gelman_rubin(chain_SS, 100, 10, 3)

# Plot the Gelman-Rubin statistic
funcs.plot_gelman_rubin(R_SS, ndim, "SS")


# Acceptance fraction
accept_frac_SS = 1
print(
    "The acceptance fraction is for the zeus sampler should be 1, as it is a slice sampler"
)

# Plot the chain to see the burn-in
funcs.plot_chain(chain_SS[:, 0, :], 1000, ndim, "SS")

burn_in = 100

# Remove burn-in
chain_SS = chain_SS[burn_in:, :, :]

# Get the autocorrelation time, tau, for the chain
taus_SS = zeus.AutoCorrTime(chain_SS)
print("Autocorrelation:", taus_SS)

tau_SS = max(taus_SS)
print("tau_SS = ", tau_SS)

# Thin the chain
chain_SS = chain_SS[:: int(tau_SS), :]
chain_SS = chain_SS.reshape(-1, 2)
chain_SS = chain_SS.reshape(-1, ndim)

# Get estimates of alpha and beta
alpha_est_SS, beta_est_SS, Intensity_0_est_SS = np.mean(chain_SS, axis=0)
alpha_sig_SS, beta_sig_SS, Intensity_0_sig_SS = np.std(chain_SS, axis=0)


print("The estimated alpha, beta, and intensity are:")
print("alpha_SS = ", alpha_est_SS, " ± ", alpha_sig_SS)
print("beta_SS = ", beta_est_SS, " ± ", beta_sig_SS)
print("Intensity_0_SS = ", Intensity_0_est_SS, " ± ", Intensity_0_sig_SS)

# Plot the distribution of the samples in a corner plot
funcs.plot_corner(chain_SS, ndim, "SS")

# ---------------------------------
# Using the emcee sampler (emcee)
# ---------------------------------
print("        ")
print("------ Emcee ------")
print("Using the emcee sampler:")
