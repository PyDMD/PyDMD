Welcome to PyDMD's documentation!
===================================================

.. image:: _static/logo_PyDMD.png
   :height: 150px
   :width: 150 px
   :align: right

Python Dynamic Mode Decomposition.


Description
^^^^^^^^^^^^

PyDMD is a Python package that uses Dynamic Mode Decomposition for a data-driven model simplification based on spatiotemporal coherent structures.


Installation
^^^^^^^^^^^^
Mac and Linux users can install pre-built binary packages using pip. To install PyDMD using pip, just type:
::

    pip install PyDMD

To install the pydmd package from source, open the terminal/command line and clone the repository with the command
::

	 git clone https://github.com/mathLab/PyDMD

After installing the dependencies you can navigate into the ``PyDMD`` folder (where the ``setup.py`` file is located) and run the command
::

	 python setup.py install

You should now be able to import the pydmd library in Python scripts and interpreters with the command ``import pydmd``.

To uninstall the package you have to rerun the installation and record the installed files in order to remove them:
::

    python setup.py install --record installed_files.txt
    cat installed_files.txt | xargs rm -rf


Developer's Guide
^^^^^^^^^^^^^^^^^^^^^^^

.. toctree::
   :maxdepth: 1

   code
   contact
   contributing
   LICENSE


Tutorials
^^^^^^^^^^

We made some tutorial examples. Please refer to the official GitHub repository for the last updates. Here the list of the exported tutorials:

- `Tutorial 1 <tutorial1dmd.html>`_ shows the typical use case and the basic features of the DMD class.
- `Tutorial 2 <tutorial2advdmd.html>`_ shows a more sophisticated application of the standard DMD algorithm.
- `Tutorial 3 <tutorial3mrdmd.html>`_ shows the possibilities of the multiresolution DMD (mrDMD) with respect to the classical DMD.




Indices and tables
^^^^^^^^^^^^^^^^^^^^^^^^

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

