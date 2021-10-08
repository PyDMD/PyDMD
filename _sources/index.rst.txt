Welcome to PyDMD's documentation!
===================================================

.. image:: _static/logo_PyDMD.png
   :height: 150px
   :width: 150 px
   :align: right

Python Dynamic Mode Decomposition.


Description
--------------------
PyDMD is a Python package that uses Dynamic Mode Decomposition for a data-driven model simplification based on spatiotemporal coherent structures.

Dynamic Mode Decomposition (DMD) is a model reduction algorithm developed by Schmid (see "Dynamic mode decomposition of numerical and experimental data"). Since then has emerged as a powerful tool for analyzing the dynamics of nonlinear systems. DMD relies only on the high-fidelity measurements, like experimental data and numerical simulations, so it is an equation-free algorithm. Its popularity is also due to the fact that it does not make any assumptions about the underlying system. See Kutz ("Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems") for a comprehensive overview of the algorithm and its connections to the Koopman-operator analysis, initiated in Koopman ("Hamiltonian systems and transformation in Hilbert space"), along with examples in computational fluid dynamics.

In the last years many variants arose, such as multiresolution DMD, compressed DMD, forward backward DMD, and higher order DMD among others, in order to deal with noisy data, big dataset, or spurius data for example.

In the PyDMD package we implemented in Python the majority of the variants mentioned above with a user friendly interface. We also provide many tutorials that show all the characteristics of the software, ranging from the basic use case to the most sofisticated one allowed by the package.

The research in the field is growing both in computational fluid dynamic and in structural mechanics, due to the equation-free nature of the model.


Installation
--------------------
PyDMD requires requires numpy, scipy, matplotlib, sphinx (for the documentation). The code is compatible with Python 2.7 and Python 3.6. It can be installed using pip or directly from the source code.

Installing via PIP
^^^^^^^^^^^^^^^^^^^^^^^^
Mac and Linux users can install pre-built binary packages using pip.
To install the package just type:
::

    pip install pydmd

To uninstall the package:
::

    pip uninstall pydmd


Installing from source
^^^^^^^^^^^^^^^^^^^^^^^^
The official distribution is on GitHub, and you can clone the repository using
::

    git clone https://github.com/mathLab/PyDMD

To install the package just type:
::

    python setup.py install

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

::

    python setup.py install --record installed_files.txt
    cat installed_files.txt | xargs rm -rf


Developer's Guide
--------------------

.. toctree::
   :maxdepth: 1

   code
   contact
   contributing
   LICENSE


Tutorials
--------------------

We made some tutorial examples. Please refer to the official GitHub repository for the last updates. Here the list of the exported tutorials:

- `Tutorial 1 <tutorial1dmd.html>`_ - Here we show a basic application of the standard dynamic mode decomposition on a simple system in order to reconstruct and analyze it.
- `Tutorial 2 <tutorial2advdmd.html>`_ - Here we show a more complex application of the standard dynamic mode decomposition on a 2D system evolving in time, focusing on the advanced settings the class provides.
- `Tutorial 3 <tutorial3mrdmd.html>`_ - Here we show the application of the multi-resolution dynamic mode decomposition on a system that contains transient time events.
- `Tutorial 4 <tutorial4cdmd.html>`_ - Here we show the application of the compressed dynamic mode decomposition in order to decrease the computational cost required by decomposition.
- `Tutorial 5 <tutorial5fbdmd.html>`_ - Here we show the forward-backward dynamic mode decomposition on a dataset coming from a fluid dynamics problem.
- `Tutorial 6 <tutorial6hodmd.html>`_ - Here we show the higher order dynamic mode decomposition applied on 1D snapshots.
- `Tutorial 7 <tutorial7dmdc.html>`_ - Here we show the dynamic mode decomposition incorporanting the effect of control, on a toy dataset.
- `Tutorial 8 <tutorial8comparison.html>`_ - Here we show the comparison between standard DMD and the optimal closed-form DMD.
- `Tutorial 9 <tutorial9spdmd.html>`_ - Here we show the sparsity-promoting DMD on a dataset coming from an heat transfer problem.
- `Developers Tutorial 1 <dev-tutorial1.html>`_ - Here we show the procedure to extending PyDMD by adding a new version of DMD.



References
--------------------
To implement the various versions of the DMD algorithm we follow these works:

- Kutz, Brunton, Brunton, Proctor. Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems. SIAM Other Titles in Applied Mathematics, 2016.
- Gavish, Donoho. The optimal hard threshold for singular values is 4/sqrt(3). IEEE Transactions on Information Theory, 2014.
- Matsumoto, Indinger. On-the-fly algorithm for Dynamic Mode Decomposition using Incremental Singular Value Decomposition and Total Least Squares. 2017.
- Hemati, Rowley, Deem, Cattafesta. De-biasing the dynamic mode decomposition for applied Koopman spectral analysis of noisy datasets. Theoretical and Computational Fluid Dynamics, 2017.
- Dawson, Hemati, Williams, Rowley. Characterizing and correcting for the effect of sensor noise in the dynamic mode decomposition. Experiments in Fluids, 2016.
- Kutz, Fu, Brunton. Multiresolution Dynamic Mode Decomposition. SIAM Journal on Applied Dynamical Systems, 2016.
- Erichson, Brunton, Kutz. Compressed dynamic mode decomposition for background modeling. Journal of Real-Time Image Processing, 2016.
- Le Clainche, Vega. Higher Order Dynamic Mode Decomposition. Journal on Applied Dynamical Systems, 2017.


Indices and tables
--------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

