import sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

import numpy as np


class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main(self.test_args)
        sys.exit(errno)

CLASSIFIERS = [
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 3 :: Only',
    'Programming Language :: Python :: 3.5',
    'Topic :: Scientific/Engineering',
    ]

setup(
    name='scintillations',
    version='0.1.0',
    description="Generate sequences of scintillations",
    author='Frederik Rietdijk',
    author_email='freddyrietdijk@fridh.nl',
    license='LICENSE',
    packages=find_packages(exclude=["tests"]),
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'streaming',
        ],
    classifiers=CLASSIFIERS,
    tests_require = [ 'pytest', 'acoustics' ],
    cmdclass = {'test': PyTest},
    )
