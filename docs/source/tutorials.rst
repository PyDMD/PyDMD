.. _tutorials:

Tutorials
=========

.. toctree::
   :hidden:

   tutorial-1-dmd


The following tutorials cover PyDMD's capabilities from basic to advanced usage.
All tutorials are available as Jupyter notebooks (``.ipynb``) and Python scripts (``.py``).


Video Tutorial Series
---------------------

New to PyDMD? Start with our video tutorial series on YouTube:

.. raw:: html

    <div style="margin: 20px 0; text-align: center;">
        <iframe width="700" height="394"
            src="https://www.youtube.com/embed/v33cL3o2Yuk"
            title="PyDMD: A Python Package for DMD"
            frameborder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowfullscreen>
        </iframe>
    </div>

    <p style="text-align: center;">
        <a href="https://www.youtube.com/watch?v=v33cL3o2Yuk" target="_blank">
            Watch directly on YouTube
        </a>
         | 
        <a href="https://github.com/PyDMD/PyDMD/blob/master/tutorials/video-tutorial-code/dmd-basic-tutorial.ipynb" target="_blank">
            Download notebook
        </a>
    </p>


User Manuals
------------

Quick guides highlighting key modules and features. Great for new users.

- **Manual 1** — The Basics of ``BOPDMD``
  [`notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/user-manual1/user-manual-bopdmd.ipynb>`_,
  `script <https://github.com/PyDMD/PyDMD/blob/master/tutorials/user-manual1/user-manual-bopdmd.py>`_]
  — ``pydmd.BOPDMD``


Basic Tutorials
---------------

- **Tutorial 1** — Analyzing real, simple data sets with PyDMD
  `webpage <tutorial-1-dmd.html>`_ |
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial1/tutorial-1-dmd.ipynb>`_
  — ``pydmd.DMD``, ``pydmd.BOPDMD``

- **Tutorial 2** — Advanced features of standard DMD
  [`html <http://pydmd.github.io/PyDMD/tutorial2advdmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial2/tutorial-2-adv-dmd.ipynb>`_]
  — ``pydmd.DMD``

- **Tutorial 3** — Multi-resolution DMD for transient phenomena
  [`html <http://pydmd.github.io/PyDMD/tutorial3mrdmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial3/tutorial-3-mrdmd.ipynb>`_]
  — ``pydmd.MrDMD``

- **Tutorial 4** — Compressed DMD for computation speedup
  [`html <http://pydmd.github.io/PyDMD/tutorial4cdmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial4/tutorial-4-cdmd.ipynb>`_]
  — ``pydmd.CDMD``

- **Tutorial 5** — Forward-backward DMD for CFD model analysis
  [`html <http://pydmd.github.io/PyDMD/tutorial5fbdmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial5/tutorial-5-fbdmd.ipynb>`_]
  — ``pydmd.FbDMD``

- **Tutorial 6** — Higher-order DMD applied to scalar time-series
  [`html <http://pydmd.github.io/PyDMD/tutorial6hodmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial6/tutorial-6-hodmd.ipynb>`_]
  — ``pydmd.HODMD``

- **Tutorial 7** — DMD with control
  [`html <http://pydmd.github.io/PyDMD/tutorial7dmdc.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial7/tutorial-7-dmdc.ipynb>`_]
  — ``pydmd.DMDC``

- **Tutorial 8** — Comparison between DMD and optimal closed-form DMD
  [`html <http://pydmd.github.io/PyDMD/tutorial8comparison.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial8/tutorial-8-comparisons.ipynb>`_]
  — ``pydmd.OptDMD``

- **Tutorial 9** — Sparsity-promoting DMD
  [`html <http://pydmd.github.io/PyDMD/tutorial9spdmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial9/tutorial-9-spdmd.ipynb>`_]
  — ``pydmd.SpDMD``

- **Tutorial 10** — Parametric DMD
  [`html <http://pydmd.github.io/PyDMD/tutorial10paramdmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial10/tutorial-10-paramdmd.ipynb>`_]
  — ``pydmd.ParametricDMD``

- **Tutorial 11** — Tikhonov regularization
  [`html <http://pydmd.github.io/PyDMD/tutorial11regularization.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial11/tutorial-11-regularization.ipynb>`_]
  — ``pydmd.DMDBase``

- **Tutorial 12** — cDMD for background modeling
  [`notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial12/tutorial-12-cdmd.ipynb>`_]
  — ``pydmd.CDMD``

- **Tutorial 13** — SubspaceDMD for locating eigenvalues of stochastic systems
  [`notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial13/tutorial-13-subspacedmd.ipynb>`_]
  — ``pydmd.SubspaceDMD``

- **Tutorial 14** — Comparison between Bagging/Optimized DMD and exact DMD
  [`html <http://pydmd.github.io/PyDMD/tutorial14-bop-dmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial14/tutorial-14-bop-dmd.ipynb>`_]
  — ``pydmd.BOPDMD``

- **Tutorial 15** — Physics-informed DMD for manifold enforcement
  [`html <http://pydmd.github.io/PyDMD/tutorial15-pidmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial15/tutorial-15-pidmd.ipynb>`_]
  — ``pydmd.PiDMD``

- **Tutorial 16** — Randomized DMD for greater computation speedup
  [`html <http://pydmd.github.io/PyDMD/tutorial16-rdmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial16/tutorial-16-rdmd.ipynb>`_]
  — ``pydmd.RDMD``

- **Tutorial 17** — Extended DMD for nonlinear eigenfunction discovery
  [`html <http://pydmd.github.io/PyDMD/tutorial17-edmd.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial17/tutorial-17-edmd.ipynb>`_]
  — ``pydmd.EDMD``

- **Tutorial 18** — LANDO for nonlinear system modeling
  [`html <http://pydmd.github.io/PyDMD/tutorial18-lando.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial18/tutorial-18-lando.ipynb>`_]
  — ``pydmd.LANDO``

- **Tutorial 19** — HAVOK for modeling chaos with partial measurements
  [`html <http://pydmd.github.io/PyDMD/tutorial19-havok.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial19/tutorial-19-havok.ipynb>`_]
  — ``pydmd.HAVOK``

- **Tutorial 20a** — COSTS for decomposing toy data
  [`notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial20/costs-tutorial_toy-data.ipynb>`_]
  — ``pydmd.COSTS``

- **Tutorial 20b** — mrCOSTS for decomposing multi-scale physics of real, noisy data
  [`notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorial20/costs-tutorial_real-data.ipynb>`_]
  — ``pydmd.mrCOSTS``

.. note::
   Tutorials 12, 13, 20a, and 20b currently only have ``.ipynb`` and ``.py`` versions
   available — rendered HTML versions are coming soon.


Developer Tutorials
-------------------

Resources for contributors looking to extend PyDMD.

- **Developer Tutorial 1** — Implementing a new version of DMD
  [`html <http://pydmd.github.io/PyDMD/dev-tutorial1.html>`_,
  `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/developers-tutorial1/developers-help-1.ipynb>`_]
  — ``pydmd.DMDBase``