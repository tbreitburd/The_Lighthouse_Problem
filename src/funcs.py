"""!@file funcs.py
@brief Module containing functions to define the distributions, prior and
posterior for the lighthouse problem, and to plot the distributions, and results

@details This module contains tools to define the lighthouse problem, and to
plot the distributions, and results. It contains 3 functions that define the
lighthouse Cauchy distribution, the log likelihood, and the log prior. It also
contains a function to plot the lighthouse Cauchy distribution.


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


def lighthouse_cauchy(alpha, beta, x):
    """!@brief Calculate the lighthouse Cauchy distribution

    @details This function takes in the parameters alpha and beta, and the
    position x, and returns the probability density function of the lighthouse
    Cauchy distribution.

    @param alpha The position of the lighthouse
    @param beta The scale of the lighthouse
    @param x The position of the flash

    @return The probability density function of the lighthouse Cauchy distribution
    """

    pdf = beta / (np.pi * (beta**2 + (x - alpha) ** 2))
    return pdf


def log_likelihood_v(param):
    """!@brief Calculate the log likelihood of the lighthouse Cauchy distribution

    @details This function takes in the parameters alpha and beta, and the
    position x, and returns the log likelihood of the lighthouse Cauchy
    distribution.

    @param param The parameters alpha and beta
    @param flash_locs The position of the flashes

    @return The log likelihood of the lighthouse Cauchy distribution
    """

    # Calculate the likelihood
    alpha, beta = param
    global flash_locs
    # likelihood = 0
    # for x in flash_locs:
    #    likelihood += (np.log(beta) - np.log(np.pi) - (np.log(beta**2 + (x - alpha)**2)))

    x = flash_locs
    epsilon = 1e-10
    likelihood = np.sum(
        [np.log(max(lighthouse_cauchy(alpha, beta, point), epsilon)) for point in x]
    )

    return likelihood


def log_prior_v(param, alpha_lim, beta_lim):
    """!@brief Calculate the log prior of the lighthouse Cauchy distribution

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

    alpha, beta = param
    global alpha_lim
    global beta_lim
    return log_likelihood_v(param) + log_prior_v(param, alpha_lim, beta_lim)


def Metroplis_Hastings(nsteps, ndim, log_posterior, cov, param_init):
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

    chain = np.zeros((nsteps, 1, ndim))  # Array to store the chain
    chain[0, 0] = param_init  # Starting point of the chain
    num_accept = 0  # Set number of accepted samples

    for i in range(nsteps - 1):
        param_current = chain[i, 0]  # Current position
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
        chain[i + 1, 0] = param_new  # Store the new position in the chain

    # Calculate the acceptance fraction
    accept_frac = num_accept / nsteps

    return chain, accept_frac


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
    project_dir = os.path.dirname(os.getcwd())
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "lighthouse_cauchy_distribution.png")
    plt.savefig(plot_path)
    plt.show()


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
    project_dir = os.path.dirname(os.getcwd())
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "lighthouse_flashes_diagram.png")
    plt.savefig(plot_path)
    plt.show()

    # Show the plot
    plt.show()


def plot_corner(chain, algorithm):
    """!@brief Plot the corner plot of the chain

    @details This function takes in the chain of samples from the chosen algorithm,
    and plots the corner plot of the chain.

    @param chain The chain of samples
    @param algorithm The chosen algorithm

    @return The corner plot of the chain
    """

    # Change chain to pandas dataframe
    chain = chain.reshape(-1, 2)
    chain = pd.DataFrame(chain, columns=["alpha", "beta"])

    # Plot the corner plot
    fig = sns.pairplot(chain, kind="hist", plot_kws={"bins": 20}, diag_kws={"bins": 20})

    # Save the plot
    project_dir = os.path.dirname(os.getcwd())
    plot_dir = os.path.join(project_dir, "Plots")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    plot_path = os.path.join(plot_dir, "corner_plot_" + algorithm + ".png")
    plt.savefig(plot_path)
    plt.show()
    return fig
