# Tutorials

In this folder we collect several useful tutorials in order to understand the principles and the potential of **PyDMD**. Please read the following table for details about the tutorials.
An additional PDF tutorial ([DSWeb contest winner](https://dsweb.siam.org/The-Magazine/All-Issues/dsweb-2019-contest-tutorials-on-dynamical-systems-software)) is available [here](tutorial_dsweb.pdf).


| Name                                                                                                                                                                                      | Description                                                        | PyDMD used classes    |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------|-----------------------|
| Tutorial1&#160;[[.ipynb](tutorial1/tutorial-1-dmd.ipynb),&#160;[.py](tutorial1/tutorial-1-dmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial1dmd.html)]                        | Analyzing real, simple data sets with PyDMD                                        | `pydmd.DMD`, `pydmd.BOPDMD`           |
| Tutorial2&#160;[[.ipynb](tutorial2/tutorial-2-adv-dmd.ipynb),&#160;[.py](tutorial2/tutorial-2-adv-dmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial2advdmd.html)]             | advanced features of standard DMD                                  | `pydmd.DMD`           |
| Tutorial3&#160;[[.ipynb](tutorial3/tutorial-3-mrdmd.ipynb),&#160;[.py](tutorial3/tutorial-3-mrdmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial3mrdmd.html)]                  | multi-resolution DMD for transient phenomena                       | `pydmd.MrDMD`         |
| Tutorial4&#160;[[.ipynb](tutorial4/tutorial-4-cdmd.ipynb),&#160;[.py](tutorial4/tutorial-4-cdmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial4cdmd.html)]                     | compress DMD for computation speedup                               | `pydmd.CDMD`          |
| Tutorial5&#160;[[.ipynb](tutorial5/tutorial-5-fbdmd.ipynb),&#160;[.py](tutorial5/tutorial-5-fbdmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial5fbdmd.html)]                  | forward-backward DMD for CFD model analysis                        | `pydmd.FbDMD`         |
| Tutorial6&#160;[[.ipynb](tutorial6/tutorial-6-hodmd.ipynb),&#160;[.py](tutorial6/tutorial-6-hodmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial6hodmd.html)]                  | higher-order DMD applied to scalar time-series                     | `pydmd.HODMD`         |
| Tutorial7&#160;[[.ipynb](tutorial7/tutorial-7-dmdc.ipynb),&#160;[.py](tutorial7/tutorial-7-dmdc.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial7dmdc.html)]                     | DMD with control                                                   | `pydmd.DMDC`          |
| Tutorial8&#160;[[.ipynb](tutorial8/tutorial-8-comparisons.ipynb),&#160;[.py](tutorial8/tutorial-8-comparisons.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial8comparison.html)] | comparison between DMD and optimal closed-form DMD                 | `pydmd.OptDMD`        |
| Tutorial9&#160;[[.ipynb](tutorial9/tutorial-9-spdmd.ipynb),&#160;[.py](tutorial9/tutorial-9-spdmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial9spdmd.html)]                  | sparsity-promoting DMD                                             | `pydmd.SpDMD`         |
| Tutorial10&#160;[[.ipynb](tutorial10/tutorial-10-paramdmd.ipynb),&#160;[.py](tutorial10/tutorial-10-paramdmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial10paramdmd.html)]   | parametric DMD                                                     | `pydmd.ParametricDMD` |
| Tutorial11&#160;[[.ipynb](tutorial10/tutorial-11-regularization.ipynb),&#160;[.py](tutorial11/tutorial-11-regularization.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial11regularization.html)]   | Tikhonov regularization                                      | `pydmd.DMDBase` |
| Tutorial12&#160;[[.ipynb](tutorial12/tutorial-12-cdmd.ipynb),&#160;[.py](tutorial12/tutorial-12-cdmd.py)]                                                                                 | cDMD for background modeling                                       | `pydmd.CDMD`          |
| Tutorial13&#160;[[.ipynb](tutorial13/tutorial-13-subspacedmd.ipynb),&#160;[.py](tutorial13/tutorial-13-subspacedmd.py)]                                                                   | SubspaceDMD for locating eigenvalues of stochastic systems         | `pydmd.SubspaceDMD`   |
| Tutorial14&#160;[[.ipynb](tutorial14/tutorial-14-bop-dmd.ipynb),&#160;[.py](tutorial14/tutorial-14-bop-dmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial14-bop-dmd.html)]     | Comparison between Bagging-/ Optimized DMD and exact DMD | `pydmd.BOPDMD`        |
| Tutorial15&#160;[[.ipynb](tutorial15/tutorial-15-pidmd.ipynb),&#160;[.py](tutorial15/tutorial-15-pidmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial15-pidmd.html)]     | Physics-informed DMD for manifold enforcement | `pydmd.PiDMD`        |
| Tutorial16&#160;[[.ipynb](tutorial16/tutorial-16-rdmd.ipynb),&#160;[.py](tutorial16/tutorial-16-rdmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial16-rdmd.html)]     | Randomized DMD for greater computation speedup | `pydmd.RDMD`        |
| Tutorial17&#160;[[.ipynb](tutorial17/tutorial-17-edmd.ipynb),&#160;[.py](tutorial17/tutorial-17-edmd.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial17-edmd.html)]     | Extended DMD for nonlinear eigenfunction discovery | `pydmd.EDMD`        |
| Tutorial18&#160;[[.ipynb](tutorial18/tutorial-18-lando.ipynb),&#160;[.py](tutorial18/tutorial-18-lando.py),&#160;[.html](http://pydmd.github.io/PyDMD/tutorial18-lando.html)]     | LANDO for nonlinear system modeling | `pydmd.LANDO`        |



# Tutorials for Developers

We collect here also the resources for helping developers to contribute to **PyDMD**.


| Name  | Description   | PyDMD used classes |
|-------|---------------|--------------------|
| Tutorial1&#160;[[.ipynb](developers-tutorial1/developers-help-1.ipynb),&#160;[.py](developers-tutorial1/developers-help-1.py),&#160;[.html](http://pydmd.github.io/PyDMD/dev-tutorial1.html)]| implementing a new version of DMD | `pydmd.DMDBase` |



#### More to come...
We plan to add more tutorials but the time is often against us. If you want to contribute with a notebook on a feature not covered yet we will be very happy and give you support on editing!
