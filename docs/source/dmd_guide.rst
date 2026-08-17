.. _dmd_guide:

DMD Variant Guide
=================

Not sure which DMD variant to use? The flowchart below will help you choose
the most appropriate method based on your data and problem type.

.. mermaid::

   flowchart TD
       START([START]) --> Q1{Is data impacted\nby a strong control\nlaw?}

       Q1 -->|Yes| BOPDMD[BOPDMDw/\nControl]
       Q1 -->|No| Q2{My data is lifted\naccording to its\nnoise structure}

       Q2 -->|Yes| Q3{My data is\nparticularly\nnoisy, statistical\nprior for noise\nis accurate}
       Q2 -->|No| Q4{Which of these\nstatements applies\nto my data?}

       Q3 -->|Yes| SPDMD[SparseCoherentSparseDMD]
       Q3 -->|No| Q5{My data is\nparticularly noisy,\nno statistical\nprior for noise\nis accurate}

       Q5 -->|Yes| RDMD[RDMD]
       Q5 -->|No| DMDwN[DMD withNoise]

       Q4 -->|My data is\nparticularly fast,\nbetween the sampling\nrate and the speed\nof the dynamics| HANKELDMD[HankelDMD]
       Q4 -->|Time-delay\npreprocessing| TIMEDMD[Time-delay\npreprocessing]

       TIMEDMD -->|Yes| CDMD[CDMD or\nDMDS]

       Q6{My data is\nparticularly slow,\nnoise structure\nfollows the\nefficiency of\nthe system} -->|Yes| PICDMD[PiDMD]
       Q6 -->|No| StandardDMD[Standard methods\nare accurate]

       Q4 --> Q6

       Q7{My data is\nfundamentally\nlow-rank} -->|Robustness\nneeded| RDMD2[RDMD]
       Q7 -->|Not needed| OPTDMD[OptDMD]

       Q4 --> Q7

       StandardDMD --> OUTCOME1[Exact DMD]
       StandardDMD --> OUTCOME2[DMD with\nControl]
       StandardDMD --> OUTCOME3[Randomized\nDMD]

   classDef default fill:#eeeeee,stroke:#000,color:#000
   classDef outcome fill:#b6e2da,stroke:#000,color:#000
   classDef special fill:#a4c2f4,stroke:#000,color:#000
   classDef warn fill:#ea9999,stroke:#000,color:#000
   classDef green fill:#b6d7a8,stroke:#000,color:#000
   classDef yellow fill:#ffe599,stroke:#000,color:#000
   classDef purple fill:#b4a7d6,stroke:#000,color:#000
   classDef pink fill:#ffcee6,stroke:#000,color:#000
   classDef orange fill:#f9cb9c,stroke:#000,color:#000
   classDef dmdblue fill:#dda6cc,stroke:#000,color:#000

   class START orange
   class BOPDMD warn
   class HANKELDMD special
   class PICDMD green
   class SPDMD yellow
   class RDMD,RDMD2 warn
   class CDMD purple
   class OPTDMD pink
   class DMDwN dmdblue
   class OUTCOME1,OUTCOME2,OUTCOME3 outcome

Guide to DMD Variants
---------------------

Noise-Robust Methods
^^^^^^^^^^^^^^^^^^^^

Use these when your data contains measurement noise:

- **Forward-Backward DMD** — corrects for sensor noise by combining forward and backward DMD.
- **Total Least-Squares DMD** — de-biases DMD for noisy datasets.
- **Optimal Closed-Form DMD** — low-rank DMD with an exact, tractable solution.
- **Subspace DMD** — stochastic Koopman analysis for noisy data.
- **Physics-Informed DMD** — incorporates known physical constraints.
- **Optimized DMD** — uses variable projection for improved accuracy.
- **BOP-DMD** — adds bagging to Optimized DMD for uncertainty quantification.

Data Compression and Sparsity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use these when working with large datasets or seeking sparse representations:

- **Compressed DMD** — reduces computational cost via random projections.
- **Randomized DMD** — efficient DMD for very large datasets.
- **Sparsity-Promoting DMD** — promotes sparse mode selection.

Including Inputs and Control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use these when your system has external inputs:

- **DMD with Control (DMDc)** — incorporates the effect of control inputs.

Transient and Multiscale Dynamics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use these for systems with multiple timescales or transient behavior:

- **Multiresolution DMD** — captures dynamics at multiple time scales.
- **Higher Order DMD** — for 1D snapshots or delay-embedded systems.

Parameterized Systems
^^^^^^^^^^^^^^^^^^^^^

Use these when your system depends on parameters:

- **Parametric DMD** — forecasts parametric dynamical systems.

Kernel and Nonlinear Methods
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use these for nonlinear systems:

- **Extended DMD** — kernel-based method for Koopman spectral analysis.
- **LANDO** — kernel learning for robust DMD with nonlinear disambiguation.
