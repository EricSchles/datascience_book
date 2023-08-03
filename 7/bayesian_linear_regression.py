import matplotlib.pyplot as plt
import numpy as np
import random

"""
Create Your Own Metropolis-Hastings Markov Chain Monte Carlo Algorithm for Bayesian Inference (With Python)
Philip Mocz (2023), @PMocz

Apply Markov Chain Monte Carlo to fit exoplanet radial velocity data and
estimate the posterior distribution of the model parameters

"""

import math
import random

def update_weights(m, b, X, Y, learning_rate):
    N = X.shape[0]
    model = m*X + b
    m_deriv = (-2 * X * (Y - model)).sum()
    b_deriv = (-2 * (Y - model)).sum()
    
    # We subtract because the derivatives point in direction of steepest ascent
    m -= (m_deriv / N) * learning_rate
    b -= (b_deriv / N) * learning_rate
    
    return m, b

def cost_function(m, b, X, y):
    N = X.shape[0]
    return (np.square(y - m*X + b)).sum()/N

def gradient_descent(X, y, init_m, init_b, learning_rate=0.001, tolerance=1e-2, max_iter=1e5):
    m = init_m
    b = init_b
    error = []
    count = 0
    while True:
        m, b = update_weights(m, b, X, y, learning_rate)
        cur_cost = cost_function(m, b, X, y)
        error.append(cur_cost)
        count += 1
        if cur_cost < tolerance:
            break
        if count > max_iter:
            break
    return m, b, error

def log_prior(theta, theta_lo, theta_hi):
    """
    Calculate the log of the priors for a set of parameters 'theta'
    We assume uniform priors bounded by 'theta_lo' and 'theta_hi'
    """
    return -np.sum(np.log(theta_hi - theta_lo))
	
def propose(theta_prev, sigma_theta, theta_lo, theta_hi):
    """
    propose a new set of parameters 'theta' given the previous value
    'theta_prev' in the Markov chain. Choose new values by adding a 
    random Gaussian peturbation with standard deviation 'sigma_theta'.
    Make sure the new proposed value is bounded between 'theta_lo'
    and 'theta_hi'
    """
    # propose a set of parameters
    theta_prop = np.random.normal(theta_prev, sigma_theta)

    # reflect proposals outside of bounds
    too_hi = theta_prop > theta_hi
    too_lo = theta_prop < theta_lo

    theta_prop[too_hi] = 2*theta_hi[too_hi] - theta_prop[too_hi]
    theta_prop[too_lo] = 2*theta_lo[too_lo] - theta_prop[too_lo]

    return theta_prop

def eval_model(theta, X, y):
    """
    Evaluate the regression_model given parameters 'theta'
    """
    m = theta[0]
    b = theta[1]
    return gradient_descent(X, y, init_m, init_b)

def log_likelihood(pred, data, errors):
    """
    Evaluate the log likelihood of a model 'pred' given the data
    """
    error_coefficient = 1.0/np.sqrt(2.0 * np.pi * (errors**2))
    loss_normalized_by_errors = -(pred - data)**2 / (2 * (errors**2))
    return np.sum(np.log(error_coefficient) + (loss_normalized_by_errors))

def log_posterior(theta, X, errors):
    """
    Evaluate the log posterior of a model parameters 'theta' given the data
    Note: since our priors are constant, we ignore adding it
    """
    pred = eval_model(theta)
    return log_likelihood(pred, data, errors)


def main():
		
    # Generate Mock Data
    N_params = 1
    m = 5
    b = 1
    
    # Carry out MCMC fitting to get best-fit parameters	
    Nburnin = 1000
    N = 8000 + Nburnin
    theta = np.zeros((N, N_params))

    theta_prev = np.random.uniform(theta_lo, theta_hi)

    for i in range(N):
        # take random step using the proposal distribution
        theta_prop = propose(theta_prev, sigma_theta, theta_lo, theta_hi)

        P_prop = log_posterior(theta_prop, X, errors)
        P_prev = log_posterior(theta_prev, X, errors)

        U = np.random.uniform(0.0, 1.0)
        r = np.min([1.0, np.exp(P_prop-P_prev)])

        if (U <= r):
            theta[i,:] = theta_prop
            theta_prev = theta_prop
        else:
            theta[i,:] = theta_prev

        # plot proposed function
        if (i % 100) == 0:
            print(theta[i,:])
            rv = eval_model(theta[i,:], tt)
            c = (1.0 - i/N)*0.5
            plt.plot(tt, rv, linewidth=0.5, color=(c,c,c))
            plt.pause(0.0001)

    # Save figure
    plt.savefig('mcmc.png',dpi=240)
    plt.show()

    # cut off burnin
    theta = theta[Nburnin:,:]

    # Plot Posteriors

    fig, ((ax0, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(nrows=2, ncols=3, figsize=(6,4), dpi=80)

    n_bins = 20
    ax0.hist(theta[:,0], n_bins, histtype='step', fill=True)
    ax0.axvline(V, color='r', linewidth=1)
    ax0.set_title('V posterior')
    ax2.hist(theta[:,2], n_bins, histtype='step', fill=True)
    ax2.axvline(K, color='r', linewidth=1)
    ax2.set_title('K posterior')
    ax3.hist(theta[:,3], n_bins, histtype='step', fill=True)
    ax3.axvline(w, color='r', linewidth=1)
    ax3.set_title('w posterior')
    ax4.hist(theta[:,4], n_bins, histtype='step', fill=True)
    ax4.axvline(e, color='r', linewidth=1)
    ax4.set_title('e posterior')
    ax5.hist(theta[:,5], n_bins, histtype='step', fill=True)
    ax5.axvline(P, color='r', linewidth=1)
    ax5.set_title('P posterior')
    ax6.hist(theta[:,6], n_bins, histtype='step', fill=True)
    ax6.axvline(chi, color='r', linewidth=1)
    ax6.set_title('chi posterior')

    fig.tight_layout()

    # Save figure
    plt.savefig('mcmc2.png',dpi=240)
    plt.show()

    return 0



if __name__== "__main__":
    main()
