"""!@file funcs.py
@brief Module containing functions to define the distributions, prior and
posterior for the lighthouse problem, the metropolis-hastings algorithm,
and to plot the distributions, and results

@details This module contains tools to define the lighthouse problem, and to
plot the distributions, and results. It contains 3 functions that define the
lighthouse Cauchy distribution, the log likelihood, and the log prior.
It also contains a function to run the Metropolis-Hastings algorithm.
Finally, it contains functions to plot the lighthouse Cauchy distribution,
the lighthouse problem, the corner plot of the chain, the chain of samples,
and the Gelman-Rubin statistic.

@author Created by T.Breitburd on 13/03/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import pandas as pd
import os

# ---------------------------------
# General functions
# ---------------------------------

path = "lighthouse_flash_data.txt"

# Get the parent directory of the current script
proj_dir = os.getcwd()
data_path = os.path.join(proj_dir, path)

# Read the data
data = np.loadtxt(data_path)
flash_locs = data[:, 0]

# Set the limits for alpha and beta
alpha_lim = 100
beta_lim = 100
intensity_0_min = 0.0001
intensity_0_max = 50


def lighthouse_cauchy(alpha, beta, x):
    """!@brief Calculate the lighthouse Cauchy distribution

    @details This function takes in the parameters alpha and beta, and the flash
    location x, and returns the probability density function of the lighthouse
    Cauchy distribution.

    @param alpha The position of the lighthouse along the coast, float
    @param beta The distance of the lighthouse to the coastline, positive float
    @param x The position of the flash, float

    @return The probability density function of the lighthouse Cauchy distribution
    """

    pdf = beta / (np.pi * (beta**2 + (x - alpha) ** 2))
    return pdf


def log_likelihood_v(param):
    """!@brief Calculate the log likelihood of the lighthouse Cauchy distribution

    @details This function takes in the parameters alpha and beta, and the
    position x, and returns the log likelihood of the flash Cauchy
    distribution as the sum of the log of the flash Cauchy likelihood.

    @param param The parameters alpha and beta, tuple
    @param flash_locs The position of the flashes, list

    @return The log likelihood of the flash Cauchy distribution
    """

    alpha, beta = param
    global flash_locs
    x = flash_locs

    # Calculate the likelihood
    epsilon = 1e-10  # Small value to avoid log(0)
    likelihood = np.sum(
        [np.log(max(lighthouse_cauchy(alpha, beta, point), epsilon)) for point in x]
    )

    return likelihood


def log_prior_v(param, alpha_lim, beta_lim):
    """!@brief Calculate the log joint prior of the lighthouse parameters

    @details This function takes in the parameters alpha and beta, and the
    limits for alpha and beta, and returns the log prior of the lighthouse
    Cauchy distribution.

    @param param The parameters alpha and beta
    @param alpha_lim The limit for alpha
    @param beta_lim The limit for beta

    @return The log prior of the lighthouse Cauchy distribution
    """

    alpha, beta = param
    # Uniform prior
    prior = -np.log(((2 * alpha_lim) * beta_lim))

    if -alpha_lim < alpha < alpha_lim and 0 < beta < beta_lim:
        return prior
    else:
        return -np.inf


def log_posterior_v(param):
    """!@brief Calculate the log posterior of the lighthouse Cauchy distribution

    @details This function takes in the parameters alpha and beta, and the
    position x, and returns the log posterior of the lighthouse Cauchy
    distribution.

    @param param The parameters alpha and beta
    @param flash_locs The position of the flashes

    @return The log posterior of the lighthouse Cauchy distribution
    """

    global alpha_lim
    global beta_lim
    return log_likelihood_v(param) + log_prior_v(param, alpha_lim, beta_lim)


def Metropolis_Hastings(nsteps, ndim, log_posterior, cov, param_init):
    """!@brief Run the Metropolis-Hastings algorithm

    @details This function takes in the number of steps, the number of dimensions,
    the log posterior, the covariance matrix, and the initial parameters, and
    returns the chain of samples and the acceptance fraction.

    @param nsteps The number of steps
    @param ndim The number of dimensions
    @param log_posterior The log posterior
    @param cov The covariance matrix
    @param param_init The initial parameters

    @return The chain of samples and the acceptance fraction
    """

    chain = np.zeros((nsteps, ndim))  # Array to store the chain
    chain[0, :] = param_init  # Starting point of the chain
    num_accept = 0  # Set number of accepted samples

    for i in range(nsteps - 1):
        param_current = chain[i]  # Current position
        Q = stats.multivariate_normal(param_current, cov)  # Poposal distribution
        param_proposed = Q.rvs()  # Proposed position
        log_a = log_posterior(param_proposed) - log_posterior(
            param_current
        )  # Acceptance ratio
        u = np.random.uniform()  # Uniform random variable
        if np.log(u) < log_a:  # Acceptance condition
            param_new = param_proposed  # ACCEPT
            num_accept += 1  # Count the numnber of accepted samples
        else:
            param_new = param_current  # REJECT, stay in the same position
        chain[i + 1] = param_new  # Store the new position in the chain

    # Calculate the acceptance fraction
    accept_frac = num_accept / nsteps

    return chain, accept_frac


def gelman_rubin(chains, window_size, step_size, ndim):
    """!@brief Calculate the Gelman-Rubin statistic

    @details This function takes a number of chains fron the same algorithm started at
    different initial values, and returns the Gelman-Rubin statistic,
    which is a measure of convergence of the MCMC algorithm, by looking at the variance
    between the chains and the variance within the chains. It uses a rolling window to calculate
    the Gelman-Rubin statistic at different points in the chain.

    @param chains The chains of samples
    @param window_size The size of the window
    @param step_size The step size
    @param ndim The number of dimensions

    @return The Gelman-Rubin statistic
    """

    # Number of samples
    nsamples = chains.shape[0]

    # Number of chains
    nchains = chains.shape[1]
    R_hat = np.zeros(((nsamples - window_size + 1) // step_size, ndim))

    # Use a rolling window to calculate the Gelman-Rubin statistic
    for idx, i in enumerate(range(0, nsamples - window_size + 1, step_size)):
        indices = tuple(range(i, i + window_size))

        # Calculate the between-chain variance
        chain_means = np.mean(chains[indices, :, :], axis=0)
        mean = np.mean(chain_means, axis=0)
        B = (window_size / (nchains - 1)) * np.sum((chain_means - mean) ** 2, axis=0)
        # Calculate the within-chain variance
        W = (
            np.sum(
                [
                    np.sum((chains[indices, i, :] - chain_means[i, :]) ** 2)
                    / (window_size - 1)
                    for i in range(nchains)
                ]
            )
            / nchains
        )

        # Calculate the Gelman-Rubin statistic
        R_hat[idx - 1, :] = (
            ((window_size - 1) / window_size) * W + (1 / window_size) * B
        ) / (W)

    return R_hat


def cauchy_clt(alpha, beta, sample_size):
    """!@brief Plot the central limit theorem for the Cauchy distribution

    @details This function takes in the parameters alpha and beta, and the
    sample size, and plots the central limit theorem for the Cauchy distribution.
    The code was obtained from Pedro Pessoa's blog:
    https://github.com/PessoaP/blog/tree/master/Lighthouse

    @param alpha The position of the lighthouse along the coast
    @param beta The distance of the lighthouse to the coastline
    @param sample_size The sample size, minimum 10,000

    @return The plot of the central limit theorem for the Cauchy distribution
    """

    # Uniformly sample theta from -pi/2 to pi/2
    theta = np.pi * (np.random.rand(sample_size) - 1 / 2)

    # Calculate the sample x
    sample = alpha + beta * np.tan(theta)  # sample x

    fig, ax = plt.subplots(1, figsize=(8, 4))
    n = [10, 31, 100, 316, 1000, 3162, 10000]
    ax.axvline(2, color="y", label="True Alpha")

    label = True
    for lim in n:
        mean = np.mean(sample[:lim])
        sig = np.std(sample[:lim]) / np.sqrt(lim)
        confMS = (mean - sig, mean + sig)
        ax.scatter(mean, lim, color="b", label="Mean")
        ax.plot(confMS, lim * np.ones(2), lw=4, color="r", alpha=0.5, label="Interval")

        if label:
            fig.legend(loc=8, bbox_to_anchor=(0.25, 0.2))
            label = False

    ax.set_title("Mean and standard deviation", fontsize=17)
    ax.set_yscale("log")
    ax.set_ylabel("# of datapoints", fontsize=16)
    ax.tick_params(axis="y", labelsize=14)
    ax.set_xlabel(r"Alpha", fontsize=15)
    fig.tight_layout()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "cauchy_CLT.png")
    plt.savefig(plot_path)
    plt.close()


def lighthouse_intensity(alpha, beta, intensity_0, x, log_intensity):
    """!@brief Calculate the lighthouse Cauchy distribution with intensity

    @details This function takes in the parameters alpha, beta and intensity, and the
    position x, and returns the probability density function of the lighthouse
    Cauchy distribution value for these.

    @param alpha The position of the lighthouse, float
    @param beta The scale of the lighthouse, positive float
    @param intensity_0 The intensity of the flash at the lighthouse, positive float
    @param x The position of the flash along the coastline, float
    @param log_intensity The log of the intensity of the observed flash, float

    @return The probability density function of the lighthouse Cauchy distribution
    """

    # Set some variables
    sigma = 1
    d_squared = ((x - alpha) ** 2) + (beta**2)
    denominator = np.sqrt(2 * np.pi * (sigma**2))

    if intensity_0 <= 0:
        return 0  # Return 0 if intensity_0 is non-positive at some point in the chain

    # Calculate the probability density function
    expo_term = -((log_intensity - np.log(intensity_0) + np.log(d_squared)) ** 2) / (
        2 * (sigma**2)
    )
    pdf = np.exp(expo_term) / denominator

    return pdf


def log_likelihood_vii(param):
    """!@brief Calculate the log likelihood of the lighthouse Cauchy distribution

    @details This function takes in the parameters alpha, beta, and intensity_0,
    and returns the log likelihood of the flash location and intensity,
    as the sum of the log of the flash Cauchy likelihood and the log of the
    intensity likelihood.

    @param param The parameters alpha, beta, and intensity_0

    @return The log likelihood of the flash Cauchy distribution
    """

    # Set some variables
    alpha, beta, intensity_0 = param
    global data
    global flash_locs
    x = flash_locs
    epsilon = 1e-10

    # Calculate the likelihood
    likelihood = np.sum(
        [np.log(max(lighthouse_cauchy(alpha, beta, point), epsilon)) for point in x]
    ) + np.sum(
        [
            np.log(
                max(
                    lighthouse_intensity(
                        alpha, beta, intensity_0, point, np.log(intensity)
                    ),
                    epsilon,
                )
            )
            for point, intensity in data
        ]
    )

    return likelihood


def log_prior_vii(param, alpha_lim, beta_lim, intensity_0_max, intensity_0_min):
    """!@brief Calculate the log prior of the lighthouse Cauchy distribution with intensity

    @details This function takes in the parameters alpha and beta, and the
    limits for alpha and beta, and returns the log prior of the lighthouse
    Cauchy distribution.

    @param param The parameters alpha and beta
    @param alpha_lim The limit for alpha
    @param beta_lim The limit for beta

    @return The log prior of the lighthouse Cauchy distribution
    """

    alpha, beta, intensity_0 = param

    if (
        -alpha_lim < alpha < alpha_lim
        and 0 < beta < beta_lim
        and intensity_0_min < intensity_0 < intensity_0_max
    ):
        # Uniform prior
        prior_alpha_beta = -np.log(((2 * alpha_lim) * beta_lim))
        prior_log_intensity = -np.log(
            intensity_0 * (np.log(intensity_0_max) - np.log(intensity_0_min))
        )

        log_prior = prior_alpha_beta + prior_log_intensity
        return log_prior
    else:
        return -np.inf


def log_posterior_vii(param):
    """!@brief Calculate the log posterior of the lighthouse parameters

    @details This function takes in the parameters alpha, beta, and I_0,
    evaluates the log likelihood of the flash location and intensity,
    as well as the log joint prior distribution of the lighthouse parameters,
    and returns the log posterior of the lighthouse parameters.

    @param param The parameters alpha, beta, and intensity_0

    @return The log posterior distribution of the lighthouse location and intensity parameters
    """

    global alpha_lim
    global beta_lim
    global intensity_0_lim
    return log_likelihood_vii(param) + log_prior_vii(
        param, alpha_lim, beta_lim, intensity_0_max, intensity_0_min
    )


# ---------------------------------
# Plotting functions
# ---------------------------------


def plot_lighthouse_cauchy(alpha, beta, x):
    """!@brief Plot the lighthouse Cauchy distribution

    @details This function takes in the parameters alpha and beta, and the
    position x, and plots the lighthouse Cauchy distribution.

    @param alpha The position of the lighthouse
    @param beta The scale of the lighthouse
    @param x The position of the flash

    @return The plot of the lighthouse Cauchy distribution
    """
    pdf = lighthouse_cauchy(alpha, beta, x)
    plt.plot(x, pdf, label=r"$\alpha = {}, \beta = {}$".format(alpha, beta))
    plt.xlabel("x")
    plt.ylabel("Probability density")
    plt.title("Lighthouse Cauchy distribution")
    plt.legend()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "lighthouse_cauchy_distribution.png")
    plt.savefig(plot_path)
    plt.close()


def plot_lighthouse(flashes, lighthouse_location):
    """!@brief Plot the lighthouse problem

    @details This function takes in the observed position of the flashes
    and a hypothetical location of the lighthouse,
    and plots the lighthouse problem.

    @param flashes The observed position of the flashes
    @param lighthouse_location The hypothetical location of the lighthouse

    @return The plot of the lighthouse problem
    """

    # Plot the sea and coast
    plt.fill_betweenx(
        y=np.linspace(-10, 10, 1000),
        x1=np.min(flashes) - 10,
        x2=np.max(flashes) + 10,
        color="orange",
        alpha=0.6,
        where=(np.linspace(-10, 10, 1000) < 0.015),
    )
    plt.fill_betweenx(
        y=np.linspace(-10, 10, 1000),
        x1=np.min(flashes) - 10,
        x2=np.max(flashes) + 10,
        color="blue",
        alpha=0.5,
        where=(np.linspace(-10, 10, 1000) > -0.01),
    )

    # Plot observed flash locations
    plt.scatter(
        flashes,
        np.zeros_like(flashes),
        marker="*",
        color="grey",
        s=90,
        label="Observed Flashes",
    )

    # Plot the lighthouse
    plt.scatter(
        lighthouse_location[0],
        lighthouse_location[1],
        marker="o",
        s=150,
        color="red",
        label="Lighthouse",
    )

    # Plot lines connecting flashes to the lighthouse location
    for flash in flashes:
        plt.plot(
            [flash, lighthouse_location[0]],
            [0, lighthouse_location[1]],
            linestyle="--",
            color="yellow",
            alpha=0.5,
        )

    plt.grid(alpha=0.3)

    # Set labels and legend
    plt.xlim([np.min(flashes) - 10, np.max(flashes) + 10])
    plt.ylim([-0.05, lighthouse_location[1] + 0.5 * lighthouse_location[1]])
    plt.xlabel("Flash Locations")
    plt.ylabel("Distance from Coastline")
    plt.legend()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "lighthouse_flashes_diagram.png")
    plt.savefig(plot_path)
    plt.close()


def plot_corner(chain, ndim, algorithm):
    """!@brief Plot the corner plot of the chain

    @details This function takes in the chain of samples from the chosen algorithm,
    and plots the corner plot of the chain.

    @param chain The chain of samples
    @param ndim The number of dimensions
    @param algorithm The chosen algorithm

    @return The corner plot of the chain
    """

    # Change chain to pandas dataframe
    chain = chain.reshape(-1, ndim)
    if ndim == 3:
        columns = ["alpha", "beta", "intensity_0"]
    else:
        columns = ["alpha", "beta"]
    chain = pd.DataFrame(chain, columns=columns)

    # Plot the corner plot
    sns.pairplot(
        chain, kind="hist", plot_kws={"bins": 20}, diag_kws={"bins": 20}, corner=True
    )

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if ndim == 3:
        plot_path = os.path.join(plot_dir, "corner_plot_vii" + algorithm + ".png")
    else:
        plot_path = os.path.join(plot_dir, "corner_plot_v" + algorithm + ".png")
    plt.savefig(plot_path)
    plt.close()


def plot_chain(chain, nsteps, ndim, algorithm):
    """!@brief Plot the chain of samples

    @details This function takes in the chain of samples from the chosen algorithm,
    the number of steps to check, and the chosen algorithm, and plots the chain of samples.

    @param chain The chain of samples
    @param nsteps The number of steps
    @param ndim The number of dimensions
    @param algorithm The chosen algorithm

    @return The chain of samples
    """

    # Plot the chain of samples
    fig, ax = plt.subplots(ndim, 1, figsize=(7, 7))
    ax[0].plot(chain[:nsteps, 0], label=r"$\alpha$")
    plt.ylabel("Parameter value")
    ax[0].legend()
    ax[1].plot(chain[:nsteps, 1], label=r"$\beta$")
    ax[1].set_xlabel("Chain Step")
    ax[1].set_ylabel("Parameter value")
    ax[1].legend()
    if ndim == 3:
        ax[2].plot(chain[:nsteps, 2], label=r"$I_{0}$")
        ax[2].set_xlabel("Chain Step")
        ax[2].set_ylabel("Parameter value")
        ax[2].legend()
    plt.suptitle("Chain of samples")
    plt.tight_layout()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if ndim == 3:
        plot_path = os.path.join(plot_dir, "chain_plot_vii" + algorithm + ".png")
    else:
        plot_path = os.path.join(plot_dir, "chain_plot_v" + algorithm + ".png")
    plt.savefig(plot_path)
    plt.close()


def plot_gelman_rubin(R_hat, ndim, algorithm):
    """!@brief Plot the Gelman-Rubin statistic

    @details This function takes in the Gelman-Rubin statistic,
    and plots the Gelman-Rubin statistic.

    @param R_hat The Gelman-Rubin statistic
    @param ndim The number of dimensions
    @param algorithm The chosen algorithm

    @return The Gelman-Rubin statistic
    """

    # Plot the Gelman-Rubin statistic
    fig, ax = plt.subplots(ndim, 1, figsize=(7, 7))
    ax[0].plot(R_hat[:, 0])
    ax[0].set_xlabel("Window")
    ax[0].set_ylabel("GR for alpha")
    ax[0].grid()
    ax[1].plot(R_hat[:, 0])
    ax[1].set_xlabel("Window")
    ax[1].set_ylabel("GR for beta")
    ax[1].grid()
    if ndim == 3:
        ax[2].plot(R_hat[:, 0])
        ax[2].set_xlabel("Window")
        ax[2].set_ylabel("GR for intensity_0")
        ax[2].grid()
    plt.suptitle("Gelman-Rubin statistic")
    plt.tight_layout()

    # Save the plot
    project_dir = os.getcwd()
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    if ndim == 3:
        plot_path = os.path.join(plot_dir, "gelman_rubin_plot_vii" + algorithm + ".png")
    else:
        plot_path = os.path.join(plot_dir, "gelman_rubin_plot_v" + algorithm + ".png")
    plt.savefig(plot_path)
    plt.close()
