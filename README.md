<p align="center">
  <a href="http://mathlab.github.io/PyDMD/" target="_blank" >
    <img alt="Python Dynamic Mode Decomposition" src="readme/logo_PyDMD.png" width="200" />
  </a>
</p>
<p align="center">
    <a href="https://doi.org/10.21105/joss.00530" target="_blank">
        <img alt="JOSS DOI" src="http://joss.theoj.org/papers/10.21105/joss.00530/status.svg">
    </a>
    <a href="https://github.com/mathLab/PyDMD/blob/master/LICENSE" target="_blank">
        <img alt="Software License" src="https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square">
    </a>
	<a href="https://badge.fury.io/py/pydmd">
		<img src="https://badge.fury.io/py/pydmd.svg" alt="PyPI version"
		height="18">
	</a>
    <a href="https://travis-ci.org/mathLab/PyDMD" target="_blank">
        <img alt="Build Status" src="https://travis-ci.org/mathLab/PyDMD.svg">
    </a>
    <a href="https://coveralls.io/github/mathLab/PyDMD" target="_blank">
        <img alt="Coverage Status" src="https://coveralls.io/repos/github/mathLab/PyDMD/badge.svg">
    </a>
    <a href="https://www.codacy.com/app/mathLab/PyDMD?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=mathLab/PyDMD&amp;utm_campaign=Badge_Grade" target="_blank">
        <img alt="Codacy Badge" src="https://api.codacy.com/project/badge/Grade/75f02cdeed684c25a273eaffb0d89880">
    </a>
</p>


**PyDMD**: Python Dynamic Mode Decomposition

## Table of contents
* [Description](#description)
* [Dependencies and installation](#dependencies-and-installation)
	* [Installing via PIP](#installing-via-pip)
	* [Installing from source](#installing-from-source)
* [Documentation](#documentation)
* [Testing](#testing)
* [Examples and Tutorials](#examples)
* [Awards](#awards)
* [How to cite](#how-to-cite)
	* [References](#references)
	* [Recent works with PyDMD](#recent-works-with-pydmd)
* [Authors and contributors](#authors-and-contributors)
* [How to contribute](#how-to-contribute)
	* [Submitting a patch](#submitting-a-patch) 
* [License](#license)

## Description
**PyDMD** is a Python package that uses **Dynamic Mode Decomposition** for a data-driven model simplification based on spatiotemporal coherent structures. 

Dynamic Mode Decomposition (DMD) is a model reduction algorithm developed by Schmid (see "Dynamic mode decomposition of numerical and experimental data"). Since then has emerged as a powerful tool for analyzing the dynamics of nonlinear systems. DMD relies only on the high-fidelity measurements, like experimental data and numerical simulations, so it is an equation-free algorithm. Its popularity is also due to the fact that it does not make any assumptions about the underlying system. See Kutz ("Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems") for a comprehensive overview of the algorithm and its connections to the Koopman-operator analysis, initiated in Koopman ("Hamiltonian systems and transformation in Hilbert space"), along with examples in computational fluid dynamics.

In the last years many variants arose, such as multiresolution DMD, compressed DMD, forward backward DMD, and higher order DMD among others, in order to deal with noisy data, big dataset, or spurius data for example. 

In PyDMD we implemented the majority of the variants mentioned above with a user friendly interface. See the [**Examples**](#examples) section below and the [**Tutorials**](tutorials/README.md) to have an idea of the potential of this package.

The research in the field is growing both in computational fluid dynamic and in structural mechanics, due to the equation-free nature of the model.



## Dependencies and installation
**PyDMD** requires requires `numpy`, `scipy`, `matplotlib`, `future`, `sphinx` (for the documentation) and `nose` (for local test). The code is compatible with Python 2.7 and Python 3.6. It can be installed using `pip` or directly from the source code.

### Installing via PIP
Mac and Linux users can install pre-built binary packages using pip.
To install the package just type: 
```bash
> pip install pydmd
```
To uninstall the package:
```bash
> pip uninstall pydmd
```

### Installing from source
The official distribution is on GitHub, and you can clone the repository using
```bash
> git clone https://github.com/mathLab/PyDMD
```

To install the package just type:
```bash
> python setup.py install
```

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:

```bash
> python setup.py install --record installed_files.txt
> cat installed_files.txt | xargs rm -rf
```

## Documentation
**PyDMD** uses [Sphinx](http://www.sphinx-doc.org/en/stable/) for code documentation. You can view the documentation online [here](http://mathlab.github.io/PyDMD/). To build the html version of the docs locally simply:

```bash
> cd docs
> make html
```

The generated html can be found in `docs/build/html`. Open up the `index.html` you find there to browse.


## Testing

We are using Travis CI for continuous intergration testing. You can check out the current status [here](https://travis-ci.org/mathLab/PyDMD).

To run tests locally (`nose` is required):

```bash
> python test.py
```

## Examples and Tutorials
You can find useful tutorials on how to use the package in the [tutorials](tutorials/README.md) folder.

Here we show a simple application (taken from [tutorial 2](tutorials/tutorial-2-adv-dmd.ipynb)): we collect few snapshots from a toy system with some noise and reconstruct the entire system evolution.
<p align="center">
<img src="readme/dmd-example.png" alt>
<em>The original snapshots used as input for the dynamic mode decomposition</em>
</p>

<p align="center">
<img src="readme/dmd-example.gif" alt></br>
<em>The system evolution reconstructed with dynamic mode decomposition</em>
</p>

## Awards

* First prize winner in **DSWeb 2019 Contest** _Tutorials on Dynamical Systems Software_ (Junior Faculty Category). You can read the winner tutorial (PDF format) in the [tutorials](tutorials/tutorial_dsweb.pdf) folder.

## How to cite
If you use this package in your publications please cite the package as follows:

Demo et al., (2018). PyDMD: Python Dynamic Mode Decomposition. Journal of Open Source Software, 3(22), 530, https://doi.org/10.21105/joss.00530

Or if you use LaTeX:
```tex
@article{demo18pydmd,
  Author = {Demo, Nicola and Tezzele, Marco and Rozza, Gianluigi},
  Title = {{PyDMD: Python Dynamic Mode Decomposition}},
  Journal = {The Journal of Open Source Software},
  Volume = {3},
  Number = {22},
  Pages = {530},
  Year = {2018},
  Doi = {https://doi.org/10.21105/joss.00530}
}
```

### References
To implement the various versions of the DMD algorithm we follow these works:

* Kutz, Brunton, Brunton, Proctor. *Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems*. SIAM Other Titles in Applied Mathematics, 2016. [[DOI](https://doi.org/10.1137/1.9781611974508)] [[bibitem](readme/Kutz2016_1.bib)].
* Gavish, Donoho. *The optimal hard threshold for singular values is 4/sqrt(3)*. IEEE Transactions on Information Theory, 2014. [[DOI](https://doi.org/10.1109/TIT.2014.2323359)] [[bibitem](readme/Gavish2014.bib)].
* Matsumoto, Indinger. *On-the-fly algorithm for Dynamic Mode Decomposition using Incremental Singular Value Decomposition and Total Least Squares*. 2017. [[arXiv](https://arxiv.org/abs/1703.11004)] [[bibitem](readme/Matsumoto2017.bib)].
* Hemati, Rowley, Deem, Cattafesta. *De-biasing the dynamic mode decomposition for applied Koopman spectral analysis of noisy datasets*. Theoretical and Computational Fluid Dynamics, 2017. [[DOI](https://doi.org/10.1007/s00162-017-0432-2)] [[bibitem](readme/Hemati2017.bib)].
* Dawson, Hemati, Williams, Rowley. *Characterizing and correcting for the effect of sensor noise in the dynamic mode decomposition*. Experiments in Fluids, 2016. [[DOI](https://doi.org/10.1007/s00348-016-2127-7)] [[bibitem](readme/Dawson2016.bib)].
* Kutz, Fu, Brunton. *Multiresolution Dynamic Mode Decomposition*. SIAM Journal on Applied Dynamical Systems, 2016. [[DOI](https://doi.org/10.1137/15M1023543)] [[bibitem](readme/Kutz2016_2.bib)].
* Erichson, Brunton, Kutz. *Compressed dynamic mode decomposition for background modeling*. Journal of Real-Time Image Processing, 2016. [[DOI](https://doi.org/10.1007/s11554-016-0655-2)] [[bibitem](readme/Erichson2016.bib)].
* Le Clainche, Vega. *Higher Order Dynamic Mode Decomposition*. Journal on Applied Dynamical Systems, 2017. [[DOI](https://doi.org/10.1137/15M1054924)] [[bibitem](readme/LeClainche2017.bib)]

### Recent works with PyDMD
Here there is a list of the scientific works involving **PyDMD** you can consult and/or cite. If you want to add one, please open a PR.

* Tezzele, Demo, Rozza. *A non-intrusive approach for proper orthogonal decomposition modal coefficients reconstruction through active subspaces*. Comptes Rendus de l'Academie des Sciences DataBEST 2019 Special Issue. [[arXiv](https://arxiv.org/abs/1907.12777)] .

* Tezzele, Demo, Rozza. *Shape Optimization through Proper Orthogonal Decomposition with Interpolation and Dynamic Mode Decomposition Enhanced by Active Subspaces*. In The Proceedings of VIII International Conference on Computational Methods in Marine Engineering, pages 122–133, 2019. [[DOI](https://congress.cimne.com/marine2019/frontal/Doc/EbookMarine2019.pdf)] [[arXiv](https://arxiv.org/abs/1905.05483)].

* Tezzele, Demo, Mola, Rozza. *An integrated data-driven computational pipeline with model order reduction for industrial and applied mathematics*. Submitted, 2018. [[arXiv](https://arxiv.org/abs/1810.12364)].

* Demo, Tezzele, Gustin, Lavini, Rozza. *Shape optimization by means of proper orthogonal decomposition and dynamic mode decomposition*. In Technology and Science for the Ships of the Future: Proceedings of NAV 2018: 19th International Conference on Ship & Maritime Research, 2018. [[DOI](https://doi.org/10.3233/978-1-61499-870-9-212)] [[arXiv](https://arxiv.org/abs/1803.07368)].

* Tezzele, Demo, Gadalla, Mola, Rozza. *Model Order Reduction by means of Active Subspaces and Dynamic Mode Decomposition for Parametric Hull Shape Design Hydrodynamics*. In Technology and Science for the Ships of the Future: Proceedings of NAV 2018: 19th International Conference on Ship & Maritime Research, 2018. [[DOI](https://doi.org/10.3233/978-1-61499-870-9-569)] [[arXiv](https://arxiv.org/abs/1803.07377)].
 
* Demo, Tezzele, Mola, Rozza. *An efficient shape parametrisation by free-form deformation enhanced by active subspace for hull hydrodynamic ship design problems in open source environment*. 2018. [[arXiv](https://arxiv.org/abs/1801.06369)].

## Authors and contributors
**PyDMD** is currently developed and mantained at [SISSA mathLab](http://mathlab.sissa.it/) by
* [Nicola Demo](mailto:demo.nicola@gmail.com)
* [Marco Tezzele](mailto:marcotez@gmail.com)

under the supervision of [Prof. Gianluigi Rozza](mailto:gianluigi.rozza@sissa.it).

Contact us by email for further information or questions about **PyDMD**, or suggest pull requests. Contributions improving either the code or the documentation are welcome!


## How to contribute
We'd love to accept your patches and contributions to this project. There are just a few small guidelines you need to follow.

### Submitting a patch

  1. It's generally best to start by opening a new issue describing the bug or
     feature you're intending to fix.  Even if you think it's relatively minor,
     it's helpful to know what people are working on.  Mention in the initial
     issue that you are planning to work on that bug or feature so that it can
     be assigned to you.

  2. Follow the normal process of [forking][] the project, and setup a new
     branch to work in.  It's important that each group of changes be done in
     separate branches in order to ensure that a pull request only includes the
     commits related to that bug or feature.

  3. To ensure properly formatted code, please make sure to use 4
     spaces to indent the code. The easy way is to run on your bash the provided
     script: ./code_formatter.sh. You should also run [pylint][] over your code.
     It's not strictly necessary that your code be completely "lint-free",
     but this will help you find common style issues.

  4. Any significant changes should almost always be accompanied by tests.  The
     project already has good test coverage, so look at some of the existing
     tests if you're unsure how to go about it. We're using [coveralls][] that
     is an invaluable tools for seeing which parts of your code aren't being
     exercised by your tests.

  5. Do your best to have [well-formed commit messages][] for each change.
     This provides consistency throughout the project, and ensures that commit
     messages are able to be formatted properly by various git tools.

  6. Finally, push the commits to your fork and submit a [pull request][]. Please,
     remember to rebase properly in order to maintain a clean, linear git history.

[forking]: https://help.github.com/articles/fork-a-repo
[pylint]: https://www.pylint.org/
[coveralls]: https://coveralls.io
[well-formed commit messages]: http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html
[pull request]: https://help.github.com/articles/creating-a-pull-request


## License

See the [LICENSE](LICENSE.rst) file for license rights and limitations (MIT).
