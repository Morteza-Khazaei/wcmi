from setuptools import setup, find_packages

def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()

def read_file(file):
   with open(file) as f:
        return f.read()
    
long_description = read_file('README.md')
version = read_file('VERSION')
requirements = read_requirements('requirements.txt')

setup(
    name = 'wcmi',
    version = version,
    author = 'Morteza Khazaei',
    author_email = 'morteza.khazaei@usherbrooke.ca',
    url = 'https://github.com/Morteza-Khazaei/wcmi',
    description = 'How to inverse water cloud model (WCM).',
    long_description_content_type = 'text/x-rst',  # If this causes a warning, upgrade your setuptools package
    long_description = long_description,
    license = 'MIT license',
    package_dir={'': 'src'},
    include_package_data=True,
    package_data={'wcmi': ['datasets/backscatter/*.csv', 'datasets/RISMA/*.csv']},
    packages = find_packages(
        where='src', 
        # exclude=['dataset']
    ),  # Don't include test directory in binary distribution
    install_requires = requirements,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ]  # Update these accordingly
)