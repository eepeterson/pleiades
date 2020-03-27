import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pleiades",
    version="0.0.1",
    author="Ethan Peterson",
    author_email="ethan.peterson@wisc.edu",
    description="A package for designing plasma physics experiments",
    long_description=long_description,
    url="https://git.doit.wisc.edu/EEPETERSON4/pleiades",
    packages=setuptools.find_packages()
)
