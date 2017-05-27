from setuptools import setup, find_packages

long_description = \
'''
A Python Communications System Design Library

This is public package that is intended to function as a communications
system design code library for use in communicaitons system simulation and design.

To install just download from the git and install

Pip:
PATH>pip install [-e] simpyle_systems
The -e installs the package in develop mode

Manual:
PATH>git clone https://github.com/jsochacki/simpyle_systems.git
Then go to the location that the package was cloned to and then run
PATH\simpyle_systems>python setup.py install [develop]
Specifying develop in place of install installs in develop mode
'''

setup(
    name='simpyle_systems',
    version='0.0.0',
    license='BSD-3 clause',
    description='A Python Communications System Design Library',
    long_description=long_description,
    author='socHACKi',
    author_email='johnsochacki@hotmail.com',
    url='https://github.com/jsochacki',
    packages = find_packages(exclude=['*test*']),
    install_requires=['numpy', 'scipy'],
    keywords = ['Linear Systens',
                'Communications Systems',
                'Microwave Design'],
)
