.. _maintainers:

Maintainer's Guide
==================

This guide is intended for PyDMD developers who need to maintain and update
the documentation website. It covers everything from local setup to adding
new content and deploying changes.

Prerequisites
-------------

Before getting started, make sure you have the following installed:

- **Git** — for version control
- **Python 3.8+** — for building the documentation
- **VS Code** — recommended editor with WSL extension if on Windows
- **Pandoc** — required for rendering Jupyter notebooks

Install Pandoc on Ubuntu/WSL:

.. code-block:: bash

    sudo apt-get install pandoc

Install the required Python packages:

.. code-block:: bash

    pip install sphinx pydata-sphinx-theme nbsphinx sphinx-design
    pip install -e .


Local Setup
-----------

**Step 1 — Fork and clone the repository:**

.. code-block:: bash

    git clone https://github.com/YOUR_USERNAME/PyDMD.git
    cd PyDMD

**Step 2 — Create a working branch:**

.. code-block:: bash

    git checkout -b your-branch-name

**Step 3 — Build the documentation locally:**

.. code-block:: bash

    cd docs
    make html

**Step 4 — Preview the site:**

.. code-block:: bash

    cd build/html
    python3 -m http.server 8000

Then open ``http://localhost:8000`` in your browser.

.. tip::
    Keep the server running in one terminal and use a second terminal for
    rebuilding. After running ``make html``, just refresh your browser to
    see changes. Use ``Ctrl+Shift+R`` for a hard refresh if changes don't
    appear.


Repository Structure
--------------------

Here's an overview of the key files and folders:

.. code-block:: text

    PyDMD/
    ├── docs/
    │   ├── source/                  # All documentation source files
    │   │   ├── conf.py              # Sphinx configuration
    │   │   ├── index.rst            # Landing page
    │   │   ├── installation.rst     # Installation page
    │   │   ├── quickstart.rst       # Quickstart guide
    │   │   ├── tutorials.rst        # Tutorials index page
    │   │   ├── dmd_guide.rst        # DMD variant guide
    │   │   ├── faq.rst              # FAQ page
    │   │   ├── references.rst       # References and contact
    │   │   ├── maintainers.rst      # This file
    │   │   ├── _static/             # Images, CSS, static files
    │   │   └── *.ipynb              # Tutorial notebooks
    │   └── build/                   # Generated HTML (do not edit)
    ├── tutorials/                   # Original tutorial notebooks
    ├── pydmd/                       # PyDMD source code
    └── .github/workflows/deploy.yml # CI/CD deployment workflow


Editing Existing Pages
----------------------

All documentation pages are written in **reStructuredText** (``.rst``).
To edit a page:

1. Open the relevant ``.rst`` file in ``docs/source/`` in VS Code
2. Make your changes
3. Rebuild with ``make html`` from the ``docs/`` folder
4. Preview at ``http://localhost:8000``
5. Commit and push your changes

Basic RST formatting:

.. code-block:: rst

    Section Title
    =============

    Subsection
    ----------

    **Bold text**, *italic text*, ``inline code``

    .. code-block:: python

        # Code block
        from pydmd import DMD

    .. note::
        This is a note box.

    .. image:: _static/image.png
       :width: 700px
       :align: center


Adding a New Page
-----------------

**Step 1 — Create the file:**

Create a new ``.rst`` file in ``docs/source/``, for example ``newpage.rst``:

.. code-block:: rst

    .. _newpage:

    New Page Title
    ==============

    Content goes here.

**Step 2 — Add it to the navigation:**

Open ``docs/source/index.rst`` and add the new page to the hidden toctree:

.. code-block:: rst

    .. toctree::
        :maxdepth: 1
        :hidden:

        installation
        quickstart
        tutorials
        newpage        <- add your page here
        dmd_guide
        ...

**Step 3 — Rebuild and preview:**

.. code-block:: bash

    cd docs
    make html


Adding a New Tutorial
---------------------

**Step 1 — Copy the notebook into the docs source folder:**

.. code-block:: bash

    cp tutorials/tutorialXX/your-notebook.ipynb docs/source/

**Step 2 — Add it to the toctree in** ``tutorials.rst``:

Open ``docs/source/tutorials.rst`` and add the notebook name (without
extension) to the hidden toctree at the top:

.. code-block:: rst

    .. toctree::
       :hidden:

       tutorial-1-dmd
       your-notebook    <- add here

**Step 3 — Add an entry in the tutorials list:**

Find the appropriate section in ``tutorials.rst`` and add:

.. code-block:: rst

    - **Tutorial XX** — Your tutorial description
      `webpage <your-notebook.html>`_ |
      `notebook <https://github.com/PyDMD/PyDMD/blob/master/tutorials/tutorialXX/your-notebook.ipynb>`_
      — ``pydmd.YourClass``

**Step 4 — Rebuild and preview:**

.. code-block:: bash

    cd docs
    make html


Updating the Binder Link
------------------------

Each tutorial page has a Binder button at the top that allows users to
run the notebook interactively. This is configured in ``conf.py``:

.. code-block:: python

    nbsphinx_prolog = """
    .. raw:: html

        <div style="margin-bottom: 20px;">
            <a href="https://mybinder.org/v2/gh/PyDMD/PyDMD/master?filepath=tutorials/..." 
               target="_blank">
                <img src="https://mybinder.org/badge_logo.svg" alt="Launch Binder"/>
            </a>
        </div>
    """

To make the Binder link point to a specific tutorial, update the
``filepath`` parameter to match the notebook's path in the repository.


Pushing Changes and Deployment
-------------------------------

Once you're happy with your changes locally:

**Step 1 — Stage and commit your changes:**

.. code-block:: bash

    git add -A
    git commit -m "Brief description of your changes"

**Step 2 — Push to GitHub:**

.. code-block:: bash

    git push origin your-branch-name

**Step 3 — Automatic deployment:**

Every push to the ``website-redesign`` branch automatically triggers
the GitHub Actions workflow, which:

1. Installs all dependencies including Pandoc
2. Builds the Sphinx documentation
3. Deploys to GitHub Pages at ``https://pydmd.github.io/PyDMD``

You can monitor the deployment at:
``https://github.com/PyDMD/PyDMD/actions``

A green checkmark means the deployment succeeded. If it fails, click
on the failed run to see the error message.

.. note::
    After deployment, do a hard refresh (``Ctrl+Shift+R``) in your
    browser if changes don't appear immediately.


Common Issues
-------------

**Build fails with "No module named X"**

Install the missing module:

.. code-block:: bash

    pip install module-name

**Build fails with "Pandoc wasn't found"**

Install Pandoc:

.. code-block:: bash

    sudo apt-get install pandoc

**Changes don't appear in the browser**

Do a hard refresh with ``Ctrl+Shift+R``, or try:

.. code-block:: bash

    cd docs
    make clean
    make html

**Math equations not rendering**

Make sure ``sphinx.ext.mathjax`` is in the extensions list in
``conf.py`` and ``sphinx.ext.imgmath`` is removed.

**Git push asks for password**

GitHub no longer accepts passwords for Git operations. Use a
Personal Access Token instead. Generate one at:
``github.com → Settings → Developer settings → Personal access tokens``


Theme and Styling
-----------------

The site uses the **PyData Sphinx Theme**, the same theme used by
NumPy, SciPy, Pandas, and other NumFOCUS projects. Theme options
are configured in ``conf.py`` under ``html_theme_options``.

For full theme documentation visit:
`pydata-sphinx-theme.readthedocs.io <https://pydata-sphinx-theme.readthedocs.io>`_