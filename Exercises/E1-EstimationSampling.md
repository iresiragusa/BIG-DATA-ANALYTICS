# Estimation and Sampling

How to estimate distribution parameters (mean and variance) from random samples and compare the efficiency of different sampling techniques.

## Objectives

- Sampling: Generate random data from known distributions.

- Parameter Estimation: Use sample statistics to approximate theoretical parameters.

- Visualization: Compare theoretical vs. estimated distributions graphically.

- Error Evaluation: Calculate the Mean Squared Error (MSE) to assess estimation quality.

### Theoretical Background

For a theoretical normal distribution $N(\mu, \sigma^2)$, we often use the following estimators based on a sample of size $n$:

Sample Mean ($\hat{\mu}$):


$$\hat{\mu} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

Sample Variance ($\hat{\sigma}^2$):


$$\hat{\sigma}^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \hat{\mu})^2$$

### Prerequisites

Ensure you have the following Python libraries installed:

```numpy```: For numerical processing.

```scipy```: For statistical functions.

```from scipy.stats import multivariate_normal```

```matplotlib```: For data visualization.

## Setting the Scenario

Define the theoretical parameters for a normal distribution that we will try to "recover" via sampling:

Theoretical Mean ($\mu$): 1.5
Theoretical Standard Deviation ($\sigma$): 0.7
Number of samples ($n$): 100

**Task 0** Import the necessary libraries and implement a utility function to plot the theoretical PDF (Probability Density Function) vs. the estimated PDF and scatter the generated samples.

## Uniform Sampling (Simple Random Sampling)

Sample data points uniformly within a range centered around the mean.

- **Task 1.1** Use a random number generator to sample $n$ points from a uniform distribution in the range $[\mu - 3\sigma, \mu + 3\sigma]$.
- **Task 1.2** Implement a function ```uniform_estimate(data, n)``` that returns the estimated $\hat{\mu}$ and $\hat{\sigma}^2$.
- **Task 1.3** Visualize the result using your plotting function and calculate the MSE.
- **Task 1.4** Run a simulation of 100 runs. In each run, generate a new set of uniform samples and store the estimates. Calculate the average of these estimates to find the Expected Value of your estimator.

## Stratified Sampling

Stratified sampling involves dividing the population into non-overlapping groups (strata) and sampling from each.

- **Task 2.1** Divide the range $[\mu - 3\sigma, \mu + 3\sigma]$ into *m* strata and decide the weight to assign to each of them.
  e.g. $m=5$ and $P_k = [0.05, 0.15, 0.60, 0.15, 0.05]$ 
- **Task 2.2** Sample a number of points per stratum $n_k$ proportional to the weights ($n_k = n \cdot P_k$).
- **Task 2.4** Calculate the mean and variance of the estimator, and plot it agaist the real distribution.

## Importance Sampling

Importance sampling uses a "proposal" distribution to sample areas of the space that are more "important" or frequent.

- **Task 3.1** Extract samples for an importance function using a uniform distribution.
- **Task 3.2** Estimate the mean and variance of the estimator, and plot it agaist the real distribution.
- **Task 3.3** Create a plot showing how the estimated mean and variance converge to the theoretical values as the number of simulations increases (e.g., from 2 to 200).
- **Task 3.4** Create a plot showing how the quality of the estimate (Mean and Variance) changes as $q$, the scaling factor in importance sampling varies from 0.2 to 1.0.
