
import numpy as np
import pystan

#---------------------------------------------
# Generate data (same as used in the notebook)

np.random.seed(42)
theta_true = (25, 0.5)
xdata = 100 * np.random.random(20)
ydata = theta_true[0] + theta_true[1] * xdata

# add scatter to points
xdata = np.random.normal(xdata, 10)
ydata = np.random.normal(ydata, 10)

#----------------------------------------------
# Create the Stan model
#  this is done by defining a string of Stan code.

fit_code = """
data {
    int<lower=0> N; // number of points
    real x[N]; // x values
    real y[N]; // y values
}

parameters {
    real alpha_perp;
    real<lower=-pi()/2, upper=pi()/2> theta;
    real log_sigma;
}

transformed parameters {
    real alpha;
    real beta;
    real sigma;
    real ymodel[N];
    
    alpha <- alpha_perp / cos(theta);
    beta <- sin(theta);
    sigma <- exp(log_sigma);
    for (j in 1:N)
    ymodel[j] <- alpha + beta * x[j];
}

model {
    y ~ normal(ymodel, sigma);
}
"""

# perform the fit
fit_data = {'N': len(xdata), 'x': xdata, 'y': ydata}
fit = pystan.stan(model_code=fit_code, data=fit_data, iter=25000, chains=4)

# extract the traces
traces = fit.extract()
pystan_trace = [traces['alpha'], traces['beta'], traces['sigma']]

# save the traces with numpy
np.save("pystan_trace.npy", pystan_trace)