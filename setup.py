#running 'pip install -e .' in the top folder looks at this script. 
# It enables one to import various python scripts throughout this project as python modules 

from setuptools import setup, find_packages

setup(name='viscnn', version='1.0', packages=find_packages())