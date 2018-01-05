from setuptools import setup

def readme():
	"""
	This function just return the content of README.md
	"""
	with open('README.md') as f:
		return f.read()

setup(name='pydmd',
	  version='0.1.0',
	  description='Python Dynamic Mode Decomposition.',
	  long_description=readme(),
	  classifiers=[
	  	'Development Status :: 5 - Production/Stable',
	  	'License :: OSI Approved :: MIT License',
	  	'Programming Language :: Python :: 2.7',
	  	'Programming Language :: Python :: 3.6',
	  	'Intended Audience :: Science/Research',
	  	'Topic :: Scientific/Engineering :: Mathematics'
	  ],
	  keywords='dynamic-mode-decomposition dmd mrdmd fbdmd cdmd',
	  url='https://github.com/mathLab/PyDMD',
	  author='Nicola Demo, Marco Tezzele',
	  author_email='demo.nicola@gmail.com, marcotez@gmail.com',
	  license='MIT',
	  packages=['pydmd'],
	  install_requires=[
	  		'future',
	  		'numpy',
	  		'scipy',
	  		'matplotlib',
	  		'Sphinx==1.4',
	  		'sphinx_rtd_theme'
	  ],
	  test_suite='nose.collector',
	  tests_require=['nose'],
	  include_package_data=True,
	  zip_safe=False)
