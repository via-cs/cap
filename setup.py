#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

try:
    with open('README.md', encoding='utf-8') as readme_file:
        readme = readme_file.read()
except IOError:
    readme = ''

def load_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

install_requires = [
    'torch>=2.0.0',
    'numpy>=1.19.0',
    'pandas>=1.2.0',
    'scikit-learn>=0.24.0',
    'matplotlib>=3.3.0',
    'tqdm>=4.50.0',
]

tests_require = [
    'pytest>=6.0.0',
    'pytest-cov>=2.10.0',
]

development_requires = [
    # general
    'pip>=21.0.0',
    'wheel>=0.37.0',
    
    # style check
    'flake8>=3.9.0',
    'isort>=5.9.0',
    'autopep8>=1.5.7',
    
    # docs
    'Sphinx>=4.0.0',
    'sphinx-rtd-theme>=1.0.0',
    
    # testing
    'coverage>=6.0.0',
]

setup(
    name='cap',
    author="Yueqiao Chen",
    author_email='christinacyq08@gmail.com',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    description="CAP: A Time Series Forecasting Framework with Multiple Models",
    entry_points={
        'console_scripts': [
            'cap=cap.__main__:main'
        ]
    },
    extras_require={
        'test': tests_require,
        'dev': development_requires + tests_require,
    },
    include_package_data=True,
    install_requires=install_requires,
    keywords=['time series', 'forecasting', 'deep learning', 'transformer', 'lstm'],
    license="MIT license",
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(include=['cap', 'cap.*']),
    python_requires='>=3.8,<3.13',
    test_suite='tests',
    tests_require=tests_require,
    url='https://github.com/via-cs/cap',
    version='0.7.1.dev0',
    zip_safe=False,
)
