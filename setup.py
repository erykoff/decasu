from setuptools import setup, find_packages

exec(open('decasu/_version.py').read())

name = 'decasu'

scripts = ['scripts/decasu_hpix_mapper.py',
           'scripts/decasu_tile_mapper.py']

setup(
    name=name,
    packages=find_packages(exclude=('tests')),
    version=__version__, # noqa
    description='DECam Survey Property Maps using healsparse',
    author='Eli Rykoff and others',
    author_email='erykoff@stanford.edu',
    url='https://github.com/erykoff/decasu',
    scripts=scripts,
)
