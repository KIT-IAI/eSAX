from setuptools import find_packages
from setuptools import setup

setup(
    name='esax',
    version='1.0.0',
    description='Energy Time Series Motif Discovery using Symbolic Aggregated Approximation (eSAX) - This Python'
                  'implementation of eSAX is based on the original eSAX implementation in R from Nicole Ludwig.'
                  'Thereby, this implementation is based on the corresponding paper',
    author='eSAX-TEAM',
    url='https://github.com/KIT-IAI/eSAX',
    license='MIT',
    packages=find_packages(),
    install_requires=['dtaidistance', 'numpy', 'matplotlib', 'pandas', 'plotnine'],
    entry_points={
        'console_scripts': [
            'esax_cli = esax.main:main',
        ],
    },
)