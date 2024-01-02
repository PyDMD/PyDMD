#!/usr/bin/env python
# coding: utf-8

# # PyDMD

# ## Tutorial 1: Dynamic Mode Decomposition on a toy dataset

# In this tutorial we will show the typical use case, applying the Dynamic Mode Decomposition on the snapshots collected during the evolution of a generic system. We present a very simple system since the main purpose of this tutorial is to show the capabilities of the algorithm and the package interface.

# First of all we import our DMD classes from the PyDMD package, we set matplotlib for the notebook, and we import numpy. We additionally import some plotting tools and some data preprocessing tools.

# In[1]:


import warnings

warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt

from pydmd import DMD, BOPDMD
from pydmd.plotter import plot_eigs, plot_summary
from pydmd.preprocessing.hankel import hankel_preprocessing


# We create the input data by summing two different functions:<br>
# $f_1(x,t) = \text{sech}(x+3)\cos(2.3t)$<br>
# $f_2(x,t) = 2\text{sech}(x)\tanh(x)\sin(2.8t)$.<br>

# In[2]:


def f1(x, t):
    return 1.0 / np.cosh(x + 3) * np.cos(2.3 * t)


def f2(x, t):
    return 2.0 / np.cosh(x) * np.tanh(x) * np.sin(2.8 * t)


nx = 65  # number of grid points along space dimension
nt = 129  # number of grid points along time dimension

# Define the space and time grid for data collection.
x = np.linspace(-5, 5, nx)
t = np.linspace(0, 4 * np.pi, nt)
xgrid, tgrid = np.meshgrid(x, t)
dt = t[1] - t[0]  # time step between each snapshot

# Data consists of 2 spatiotemporal signals.
X1 = f1(xgrid, tgrid)
X2 = f2(xgrid, tgrid)
X = X1 + X2

# Make a version of the data with noise.
mean = 0
std_dev = 0.2
random_matrix = np.random.normal(mean, std_dev, size=(nt, nx))
Xn = X + random_matrix

X.shape


# The plots below represent these functions and the dataset *without* noise.

# In[3]:


titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f$"]
data = [X1, X2, X]

fig = plt.figure(figsize=(17, 6), dpi=200)
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
    plt.xlabel("Space")
    plt.ylabel("Time")
plt.colorbar()
plt.show()


# The plots below represent these functions and the dataset *with* noise.

# In[4]:


titles = ["$f_1(x,t)$", "$f_2(x,t)$", "$f$ (noisy)"]
data = [X1, X2, Xn]

fig = plt.figure(figsize=(17, 6), dpi=200)
for n, title, d in zip(range(131, 134), titles, data):
    plt.subplot(n)
    plt.pcolor(xgrid, tgrid, d.real)
    plt.title(title)
    plt.xlabel("Space")
    plt.ylabel("Time")
plt.colorbar()
plt.show()


# ## DMD steps for handling real data (i.e. data with noise)
#
# Step 1: Do a time-delay embedding (`d` is number of delay embeddings).
#
# Step 2: Apply BOP-DMD (`num_trials` is number of statistical bags).
#
# Step 3: OPTIONAL -- Constrain the eigenvalues (i) left-half plane, (ii) imaginary axis, (iii) complex conjugate pairs.

# ## Steps 1 and 2:
#
# We currently have the temporal snapshots in the input matrix rows. We can easily create a new DMD instance and exploit it in order to compute DMD on the data. Since the snapshots must be arranged by columns, we need to transpose the data matrix in this case.
#
# Starting with Step 1, we apply a time-delay embedding to our data before applying our DMD method of choice. In order to do that, we wrap our DMD instance in the `hankel_preprocessing` routine and provide our desired number of delays `d`. We will dive more into *why* we need the time-delay embedding later in the tutorial.
#
# Continuing on to Step 2, we note that in order to apply the BOP-DMD method in particular, all we need to do is build `BOPDMD` model as our particular DMD instance. Once the instance is wrapped, we can go ahead with the fit.
#
# A summary of the DMD results can then be plotted using the `plot_summary` function.

# In[5]:


# Build the Optimized DMD model.
# num_trials=0 gives Optimized DMD, without bagging.
optdmd = BOPDMD(svd_rank=4, num_trials=0)

# Wrap the model with the preprocessing routine.
delays = 2
delay_optdmd = hankel_preprocessing(optdmd, d=delays)

# Fit the model to the noisy data.
# Note: BOPDMD models need the data X and the times of data collection t for fitting.
# Hence if we apply time-delay, we must adjust the length of our time vector accordingly.
num_t = len(t) - delays + 1
delay_optdmd.fit(Xn.T, t=t[:num_t])

# Plot a summary of the DMD results.
plot_summary(delay_optdmd, d=delays)

# Print computed eigenvalues (frequencies are given by imaginary components).
# Also plot the resulting data reconstruction.
print(
    f"Frequencies (imaginary component): {np.round(delay_optdmd.eigs, decimals=3)}"
)
plt.title("Reconstructed Data")
plt.imshow(delay_optdmd.reconstructed_data.real)
plt.show()
plt.title("Ground Truth Data")
plt.imshow(X.T)
plt.show()


# The DMD object contains the principal information about the decomposition:
# - the attribute `modes` is a 2D numpy array where the columns are the low-rank structures individuated;
# - the attribute `dynamics` is a 2D numpy array where the rows refer to the time evolution of each mode;
# - the attribute `eigs` refers to the eigenvalues of the low dimensional operator;
# - the attribute `amplitudes` gives the spatiotemporal mode coefficients used for reconstruction;
# - the attribute `reconstructed_data` refers to the approximated system evolution.
#
# Although these attributes may be accessed directly from a fitted DMD object as demonstrated below, we note that the `plot_summary` function plots a summarizing view of many of these attributes automatically.

# In[6]:


colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# Plotting the modes individually...
plt.figure(figsize=(14, 3))
for i, mode in enumerate(delay_optdmd.modes.T):
    # Get the average across delays, since we used time-delay.
    mode = np.average(mode.reshape(delays, len(mode) // delays), axis=0)
    plt.subplot(1, len(delay_optdmd.modes.T), i + 1)
    plt.plot(mode.real, c=colors[i])
    plt.title(f"Mode {i + 1}")
plt.tight_layout()
plt.show()

# Plotting the dynamics individually...
plt.figure(figsize=(14, 3))
for i, dynamic in enumerate(delay_optdmd.dynamics):
    plt.subplot(1, len(delay_optdmd.dynamics), i + 1)
    plt.plot(t[:num_t], dynamic.real, c=colors[i])
    plt.title(f"Dynamics {i + 1}")
plt.tight_layout()
plt.show()

# Plot the eigenvalues.
plot_eigs(delay_optdmd, show_axes=True, show_unit_circle=False, figsize=(4, 4))

# Print the amplitudes.
print(f"Computed amplitudes: {np.round(delay_optdmd.amplitudes, decimals=3)}\n")


# ## [Optional] Step 3: DMD with constraints
#
# `BOPDMD` models also have the option to specify the structure of the eigenvalues that they compute. More specifically, users can impose the following constraints, as well as any valid combination of them.
#
# - Stable: constrain eigenvalues to have non-positive real parts.
# - Imaginary: constrain eigenvalues to be purely imaginary.
# - Conjugate pairs: constrain eigenvalues to always appear with their complex conjugate.
#
# This can be especially helpful for dealing with noise and preventing growth/decay of your dynamics.

# In[7]:


# CONSTRAINTS

# Stable: constrain to the left-half plane (no positive real parts to eigenvalues).
# bopdmd = BOPDMD(eig_constraints={"stable"})

# Imaginary: constrain to imaginary axis (no real parts to eigenvalues).
# bopdmd = BOPDMD(eig_constraints={"imag"})

# Stable + Conjugate: constrain to the left-half plane and as complex conjugates.
# bopdmd = BOPDMD(eig_constraints={"stable", "conjugate_pairs"})

# Imaginary + Conjugate: constrain to imaginary axis and as complex conjugates.
# bopdmd = BOPDMD(eig_constraints={"imag", "conjugate_pairs"})

optdmd = BOPDMD(
    svd_rank=4, num_trials=0, eig_constraints={"imag", "conjugate_pairs"}
)
delay_optdmd = hankel_preprocessing(optdmd, d=delays)
delay_optdmd.fit(Xn.T, t=t[:num_t])
plot_summary(delay_optdmd, d=delays)

print(
    f"Frequencies (imaginary component): {np.round(delay_optdmd.eigs, decimals=3)}"
)
plt.title("Reconstructed Data")
plt.imshow(delay_optdmd.reconstructed_data.real)
plt.show()
plt.title("Ground Truth Data")
plt.imshow(X.T)
plt.show()


# ## Why do we use BOP-DMD?
#
# Put simply, **BOP-DMD is extremely robust to measurement noise, hence making it the preferred method when dealing with real-world data.** By contrast, the results of exact DMD (which is implemented by the `DMD` module) are extremely sensitive to measurement noise, as we demonstrate here. Note the decay of the dynamics onset by the bias in the eigenvalues. Also note how when we previously performed this fit but with BOP-DMD instead, we did not observe such decay, but rather we recovered the true oscillations.

# ### This is what happens when we use exact DMD instead of BOP-DMD:

# In[8]:


dmd = DMD(svd_rank=4)
delay_dmd = hankel_preprocessing(dmd, d=delays)
delay_dmd.fit(Xn.T)
plot_summary(delay_dmd, d=delays)

print(
    f"Frequencies (imaginary component): {np.round(np.log(delay_dmd.eigs) / dt, decimals=3)}"
)
plt.title("Reconstructed Data")
plt.imshow(delay_dmd.reconstructed_data.real)
plt.show()
plt.title("Ground Truth Data")
plt.imshow(X.T)
plt.show()


# ## Why do we need time-delay?
#
# Notice that by construction, our data set is completely real (i.e. it doesn't possess imaginary components) and it contains 2 distinct spatiotemporal features that oscillate in time. To capture such oscillations from real data sets, we need 2 DMD eigenvalues for each oscillation: one to capture the frequency of the oscillation and one to capture its complex conjugate. Hence for our particular data set, we need at least 4 DMD eigenvalues / modes in order to capture the full extent of our data. You may have noticed this as we consistently used `svd_rank=4`.
#
# However, **because our data is real *and* because the underlying spatial modes are stationary, we cannot always obtain correct results if we apply DMD directly to our data set, even if we use the proper rank truncation.** Time-delay helps mitigate this by giving us more observations to work with. As you will see below, our clean data reveals 2 dominant singular values, but applying any number of time-delay embeddings will lift this number of singular values from 2 to 4, hence allowing us to more-consistently extract the rank-4 structure that we would expect. This is also why we use `d=2` -- any number of delays greater than 1 suffices.
#
# Note that this preprocessing step may or may not be necessary depending on your particular data set. Hence the most practical thing to do during any DMD application is to **examine the singular value spectrum of you data as you apply time-delay embeddings.**

# ### This is what happens without time-delay (using clean data and exact DMD):

# In[9]:


dmd = DMD(svd_rank=4)
dmd.fit(X.T)
plot_summary(dmd)

print(
    f"Frequencies (imaginary component): {np.round(np.log(dmd.eigs) / dt, decimals=3)}"
)
plt.title("Reconstructed Data")
plt.imshow(dmd.reconstructed_data.real)
plt.show()
plt.title("Ground Truth Data")
plt.imshow(X.T)
plt.show()


# ### This is what happens with time-delay (using clean data and exact DMD):

# In[10]:


dmd = DMD(svd_rank=4)
delay_dmd = hankel_preprocessing(dmd, d=2)
delay_dmd.fit(X.T)
plot_summary(delay_dmd, d=2)

print(
    f"Frequencies (imaginary component): {np.round(np.log(delay_dmd.eigs) / dt, decimals=3)}"
)
plt.title("Reconstructed Data")
plt.imshow(delay_dmd.reconstructed_data.real)
plt.show()
plt.title("Ground Truth Data")
plt.imshow(X.T)
plt.show()


# In[ ]:
