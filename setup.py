#!/usr/bin/env python

import distutils.core
distutils.core.setup(
    name='segment',
    version='0.1',
    url='http://rmozone.com/',
    description='a/v segmenter/analyzer',
    author='Robert M Ochshorn',
    author_email='mail@RMOZONE.COM',
    packages=['segment'],
    scripts=['bin/segment'],
    classifiers=[
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Programming Language :: Python",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Artistic Software",
        "Topic :: Multimedia",
        "Topic :: Scientific/Engineering",
        ],
    keywords='numerical media analysis segmentation audio video numpy',
    license='GPL')
