import numpy as np
import zeus
import os
import funcs

# Ignore seaborn FutureWarnings
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


print("---------------------------------")
print("----------- Part V --------------")
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

print("---------------------------------")
print("---------------------------------")
print("(a) The Data")
print("The observed flashes have locations:")
print(flash_locs)

# Plot the data as a diagram of the lighthouse and the flashes
# for a hypothetical lighthouse at position alpha = 0 and beta = 0.1
funcs.plot_lighthouse(flash_locs, (0, 0.1))

# Visualise the Cauchy distribution for the lighthouse problem,
# again for hypothetical alpha = 0 and beta = 0.1
x = np.linspace(-10, 10, 1000)
funcs.plot_lighthouse_cauchy(0, 0.1, x)

# --------------------------------------------------------------
# --------------------------------------------------------------
# Sample from the posterior
# --------------------------------------------------------------
# --------------------------------------------------------------
print("---------------------------------")
print("---------------------------------")
print("(b) Sampling from the posterior")

# ---------------------------------
# Using Metropolis-Hastings
# ---------------------------------
print("------ Metropolis-Hastings ------")
print(
    "Using the Metropolis-Hastings algorithm,"
    + " with a multivariate normal proposal distribution:"
)


nsteps, ndim = 100000, 2  # Define the number of steps and the number of dimensions
init = np.ones(ndim)  # Starting point of the chain
cov = np.array([[1.5, 0.0], [0.0, 1.5]])

# Run the Metropolis-Hastings algorithm
chain_MH, accept_frac_MH = funcs.Metroplis_Hastings(
    nsteps, ndim, funcs.log_posterior_v, cov, init
)

print("The acceptance fraction is for the Metropolis Hastings algorithm is:")
print(accept_frac_MH)

# Get the autocorrelation time, tau, for the chain
taus_MH = zeus.AutoCorrTime(chain_MH)
print("Autocorrelation:", taus_MH)

tau_MH = max(taus_MH)
print("tau_MH = ", tau_MH)

# Plot the distribution of the samples in a corner plot
funcs.plot_corner(chain_MH, "MH")

# Get estimates of alpha and beta
alpha_est_MH, beta_est_MH = np.mean(chain_MH, axis=0)[0]
alpha_sig_MH, beta_sig_MH = np.std(chain_MH, axis=0)[0]

print("The estimated alpha and beta are:")
print("alpha_MH = ", alpha_est_MH, " ± ", alpha_sig_MH)
print("beta_MH = ", beta_est_MH, " ± ", beta_sig_MH)

# ---------------------------------
# Using the zeus sampler (ensemble sampler)
# ---------------------------------

print("------ Zeus Slice Sampling ------")
print("Using the zeus sampler, with slice sampling:")

nsteps, nwalkers, ndim = 10000, 10, 2
start = np.abs(np.random.randn(nwalkers, ndim))

sampler_SS = zeus.EnsembleSampler(nwalkers, ndim, funcs.log_posterior_v)

sampler_SS.run_mcmc(start, nsteps)
chain_SS = sampler_SS.get_chain(flat=True)

# Acceptance fraction
accept_frac_SS = np.mean(sampler_SS.acceptance_fraction)
print("The acceptance fraction is for the zeus sampler is:")
print(accept_frac_SS)

# Get the autocorrelation time, tau, for the chain
taus_SS = zeus.AutoCorrTime(sampler_SS.get_chain())
print("Autocorrelation:", taus_SS)

tau_SS = max(taus_SS)
print("tau_SS = ", tau_SS)

# Plot the distribution of the samples in a corner plot
funcs.plot_corner(chain_SS, "SS")

# ---------------------------------
# Using the emcee sampler (emcee)
# ---------------------------------

print("------ Emcee ------")
print("Using the emcee sampler:")
