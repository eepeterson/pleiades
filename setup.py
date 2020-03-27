from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pleiades",
    version="0.1.0",
    author="Ethan Peterson",
    author_email="ethan.peterson@wisc.edu",
    description="A package for designing plasma physics experiments",
    long_description=long_description,
    url="https://git.doit.wisc.edu/EEPETERSON4/pleiades",
    packages=setuptools.find_packages()
)

# Get version information from __init__.py. This is ugly, but more reliable than
# using an import.
with open('pleiades/__init__.py', 'r') as f:
    version = f.readlines()[-1].split()[-1].strip("'")

kwargs = {
    'name': 'pleiades',
    'version': version,
    'packages': find_packages(exclude=['tests*']),

    # Metadata
    'author': 'Ethan Peterson',
    'author_email': 'peterson@psfc.mit.edu',
    'description': 'Pleiades',
    'project_urls': {
        'Issue Tracker': 'https://github.com/eepeterson/pleiades/issues',
        'Documentation': 'https://pleiades.readthedocs.io',
        'Source Code': 'https://github.com/eepeterson/pleiades',
    },
    'classifiers': [
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    # Dependencies
    'python_requires': '>=3.5',
    'install_requires': [
        'numpy>=1.9', 'scipy', 'ipython', 'matplotlib',
    ],
    'extras_require': {
        'test': ['pytest', 'pytest-cov', 'colorama'],
        'vtk': ['vtk'],
    },
}

setup(**kwargs)
