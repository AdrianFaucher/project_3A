import numpy as np
import matplotlib.pyplot as plt
import emcee

# Generate synthetic data
np.random.seed(42)
N = 50
x = np.sort(np.random.rand(N) * 10)
true_m = 2.5
true_b = 1.0
yerr = 1.0
y = true_m * x + true_b + np.random.normal(0, yerr, size=N)

# Plot the data
plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Synthetic linear data")
plt.legend()
plt.show()

# Define the log-likelihood
def log_likelihood(theta, x, y, yerr):
    m, b = theta
    model = m * x + b
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(2 * np.pi * sigma2))

# Define the prior (uniform priors for m and b)
def log_prior(theta):
    m, b = theta
    if -10.0 < m < 10.0 and -10.0 < b < 10.0:
        return 0.0
    return -np.inf

# Posterior = prior + likelihood
def log_posterior(theta, x, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, yerr)

# Set up the MCMC sampler
ndim = 2  # m and b
nwalkers = 50
initial = np.array([1.0, 0.0])  # initial guess
pos = initial + 1e-4 * np.random.randn(nwalkers, ndim)

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, yerr))
sampler.run_mcmc(pos, 2000, progress=True)

# Analyze the samples
samples = sampler.get_chain(discard=500, thin=15, flat=True)
m_mcmc, b_mcmc = np.mean(samples, axis=0)
print(f"Estimated m = {m_mcmc:.2f}, b = {b_mcmc:.2f}")

# Plotting posterior predictive
plt.errorbar(x, y, yerr=yerr, fmt=".k", label="Data")
x_fit = np.linspace(0, 10, 100)
for m, b in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(x_fit, m * x_fit + b, color="gray", alpha=0.1)
plt.plot(x_fit, m_mcmc * x_fit + b_mcmc, color="red", label="Best Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Posterior predictive lines")
plt.show()

import corner

# labels for the parameters
labels = ["m", "b"]

# make corner plot
fig = corner.corner(samples, labels=labels, truths=[true_m, true_b])
plt.show()
