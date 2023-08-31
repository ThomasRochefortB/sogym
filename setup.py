import pathlib
import setuptools
from setuptools import find_namespace_packages


setuptools.setup(
    name='sogym_v2',
    version='0.0.1',
    description= 'v2 of the SoGym environment for structural optimization research',
    author='Danijar Hafner',
    url='https://github.com/ThomasRochefortB/sogym_v2',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=find_namespace_packages(exclude=['example.py']),
    include_package_data=True,
    install_requires=pathlib.Path('requirements.txt').read_text().splitlines(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)