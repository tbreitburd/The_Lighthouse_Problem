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
    plt.show()
