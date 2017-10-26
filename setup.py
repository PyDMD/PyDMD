from setuptools import setup

def readme():
	"""
	This function just return the content of README.md
	"""
	with open('README.md') as f:
		return f.read()

setup(name='pydmd',
	  version='0.0.2',
	  description='Python Dynamic Mode Decomposition.',
	  long_description=readme(),
	  classifiers=[
	  	'Development Status :: 3 - Alpha',
	  	'License :: OSI Approved :: MIT License',
	  	'Programming Language :: Python :: 2.7',
	  	'Programming Language :: Python :: 3.6',
	  	'Intended Audience :: Science/Research',
	  	'Topic :: Scientific/Engineering :: Mathematics'
	  ],
	  keywords='dynamic-mode-decomposition dmd mrdmd fbdmd',
	  url='https://github.com/mathLab/PyDMD',
	  author='Marco Tezzele, Nicola Demo',
	  author_email='marcotez@gmail.com, demo.nicola@gmail.com',
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
