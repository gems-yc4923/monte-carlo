from setuptools import setup

with open('requirements.txt','r') as requirements:
    packs = requirements.readlines()

setup(
    name='mcsim',  # Package name must be mcsim.
    version='1.0.0',  # Version number, required
    packages=['mcsim'],  # directories to install, required
    install_requires = packs,
    # One-line description or tagline of what your project does
    description='A mcsim package that will help us implement a Skyrmion',  # Optional
    author='Yassine Charouif',  # Optional
    author_email='yc4923@imperial.ac.uk'  # Optional
)

