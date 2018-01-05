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

