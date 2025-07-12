from setuptools import setup, find_packages

   
"""
Run setup tool to install minimum requirements 
for the core project package
""" 
setup(
    name='secondsight',
    description='EnigmaAI core library',
    author='Anna Huang',
    packages=find_packages(where='api/src'),
    package_dir={'': 'api/src'},
    install_requires=[
        'pyyaml'
    ],
    python_requires='>=3.9',
)
