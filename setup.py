from setuptools import setup

def readme():
	"""
	This function just return the content of README.md
	"""
	with open('README.md') as f:
		return f.read()

setup(name='dmd',
	  version='0.0.1',
	  description='Dynamic Mode Decomposition.',
	  long_description=readme(),
	  classifiers=[
	  	'Development Status :: 1 - Planning',
	  	'License :: OSI Approved :: MIT License',
	  	'Programming Language :: Python :: 2.7',
	  	'Intended Audience :: Science/Research',
	  	'Topic :: Scientific/Engineering :: Mathematics'
	  ],
	  keywords='dynamic_mode_decomposition ',
	  url='https://github.com/mathLab/DMD',
	  author='Marco Tezzele, Nicola Demo',
	  author_email='marcotez@gmail.com, demo.nicola@gmail.com',
	  license='MIT',
	  packages=['dmd'],
	  install_requires=[
	  		'numpy',
	  		'scipy',
	  		'matplotlib',
	  		'enum34',
	  		'Sphinx>=1.4',
	  		'sphinx_rtd_theme'
	  ],
	  test_suite='nose.collector',
	  tests_require=['nose'],
	  include_package_data=True,
	  zip_safe=False)
