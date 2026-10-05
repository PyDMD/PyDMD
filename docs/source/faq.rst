.. _faq:

FAQ
===========================

General
-------

**What is Dynamic Mode Decomposition?**

Dynamic Mode Decomposition (DMD) is a data-driven method for analyzing
time-varying datasets. It extracts spatiotemporal coherent structures —
called DMD modes — that describe the dominant dynamics of a system. DMD
is closely related to Koopman operator theory and has applications in
fluid dynamics, neuroscience, finance, climate modeling, and more.

**Which DMD variant should I use?**

See the :doc:`dmd_guide` page for a flowchart and detailed descriptions
of all available variants. As a general rule:

- For clean data with no noise → standard ``DMD``
- For noisy data → ``BOPDMD``, ``TLSqrDMD``, or ``FBDMDOperator``
- For large datasets → ``CDMD`` or ``RDMD``
- For systems with control inputs → ``DMDc``
- For multiple timescales → ``MrDMD``
- For nonlinear systems → ``EDMD`` or ``LANDO``

**What does** ``svd_rank`` **do?**

``svd_rank`` controls the truncation of the Singular Value Decomposition
(SVD) used internally by DMD. Setting it to an integer ``r`` keeps only
the top ``r`` singular values. Setting it to ``0`` triggers an automatic
rank selection using the optimal hard threshold method. Setting it to
``-1`` keeps all singular values (no truncation).

Installation
------------

**What are PyDMD's dependencies?**

The core features require ``numpy`` and ``scipy``. For plotting you also
need ``matplotlib``. Some advanced modules require additional packages
such as ``scikit-learn`` (for kernel methods) and ``torch`` (for certain
optimized variants).

**How do I install PyDMD?**

.. code-block:: bash

    pip install pydmd

Or from source:

.. code-block:: bash

    git clone https://github.com/PyDMD/PyDMD
    cd PyDMD
    pip install -e .

Usage
-----

**How do I fit a DMD model?**

All PyDMD models follow the same interface — initialize, then call
``fit()`` with your data array:

.. code-block:: python

    from pydmd import DMD

    dmd = DMD(svd_rank=0)  # svd_rank=0 for automatic rank selection
    dmd.fit(X)  # X is an (n, m) array of m snapshots of dimension n

**How do I reconstruct my data after fitting?**

.. code-block:: python

    # Access the reconstructed data
    X_reconstructed = dmd.reconstructed_data

**How do I access DMD modes, eigenvalues, and dynamics?**

.. code-block:: python

    modes = dmd.modes        # Spatial modes
    eigs = dmd.eigs          # Eigenvalues
    dynamics = dmd.dynamics  # Temporal dynamics

**My reconstruction looks poor — what should I try?**

- Try increasing ``svd_rank`` to retain more modes
- If your data is noisy, switch to a noise-robust variant like ``BOPDMD``
- Try centering your data first using ``zero_mean_preprocessing``
- Check that your snapshots are evenly spaced in time

**How do I use preprocessors?**

.. code-block:: python

    from pydmd import DMD
    from pydmd.preprocessing import zero_mean_preprocessing

    dmd = zero_mean_preprocessing(DMD(svd_rank=12))
    dmd.fit(X)

Contributing
------------

**How do I contribute to PyDMD?**

See the :doc:`contributing` page for full details. In brief: fork the
repository, create a branch, make your changes, add tests, and open a
Pull Request on GitHub.

**How do I add a new DMD variant?**

See `Developer Tutorial 1 <tutorial1dmd.html>`_ which walks through the
full process of extending PyDMD with a new DMD implementation.