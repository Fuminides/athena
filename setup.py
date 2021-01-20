from setuptools import setup


# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf8') as f:
    long_description = f.read()


setup(
   name='Athena',
   version='0.1',
   description='BCI architectures library',
   author='Javier Fumanal Idocin',
   url='https://github.com/Fuminides/athena',
   author_email='javier.fumanal@unavarra.es',
   packages=['athena'],  #same as name
   install_requires=['numpy', 'Fancy_aggregations', 'torch', 'tensorflow', 'pandas', 'mne'], #external packages as dependencies
   long_description=long_description,
   long_description_content_type='text/markdown'
)