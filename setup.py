from setuptools import setup, find_packages

meta = {}
with open("pydmd/meta.py") as fp:
    exec(fp.read(), meta)

# Package meta-data.
NAME = meta["__title__"]
DESCRIPTION = "Python Dynamic Mode Decomposition."
URL = "https://github.com/mathLab/PyDMD"
MAIL = meta["__mail__"]
AUTHOR = meta["__author__"]
VERSION = meta["__version__"]
KEYWORDS = "dynamic-mode-decomposition dmd"

REQUIRED = ["numpy<2", "scipy", "matplotlib", "scikit-learn"]

EXTRAS = {
    "docs": ["Sphinx>=1.4", "sphinx_rtd_theme"],
    "test": ["pytest", "pytest-cov", "pytest-mock", "ezyrb>=v1.2.1.post2205"],
}

LDESCRIPTION = (
    "PyDMD is a Python package that uses Dynamic Mode Decomposition for "
    "a data-driven model simplification based on spatiotemporal coherent "
    "structures.\n"
    "\n"
    "Dynamic Mode Decomposition (DMD) is a model reduction algorithm "
    "developed by Schmid (see 'Dynamic mode decomposition of numerical and "
    "experimental data').  Since then has emerged as a powerful tool for "
    "analyzing the dynamics of nonlinear systems. DMD relies only on the "
    "high-fidelity measurements, like experimental data and numerical "
    "simulations, so it is an equation-free algorithm. Its popularity is "
    "also due to the fact that it does not make any assumptions about the "
    "underlying system. See Kutz ('Dynamic Mode Decomposition: "
    "Data-Driven Modeling of Complex Systems') for a comprehensive "
    "overview of the algorithm and its connections to the Koopman-operator "
    "analysis, initiated in Koopman ('Hamiltonian systems and "
    "transformation in Hilbert space'), along with examples in "
    "computational fluid dynamics.\n"
    "\n"
    "In the last years many variants arose, such as multiresolution DMD, "
    "compressed DMD, forward backward DMD, and higher order DMD among "
    "others, in order to deal with noisy data, big dataset, or spurius "
    "data for example.\n"
    "\n"
    "In PyDMD we implemented the majority of the variants mentioned above "
    "with a user friendly interface.\n"
    "\n"
    "The research in the field is growing both in computational fluid "
    "dynamic and in structural mechanics, due to the equation-free nature "
    "of the model.\n"
)

setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LDESCRIPTION,
    author=AUTHOR,
    author_email=MAIL,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    keywords=KEYWORDS,
    url=URL,
    license="MIT",
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    zip_safe=False,
)
