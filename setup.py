from setuptools import setup, find_packages


name = 'decasu'

scripts = ['scripts/decasu_hpix_mapper.py',
           'scripts/decasu_tile_mapper.py']

setup(
    name=name,
    packages=find_packages(exclude=('tests')),
    description='DECam Survey Property Maps using healsparse',
    author='Eli Rykoff and others',
    author_email='erykoff@stanford.edu',
    url='https://github.com/erykoff/decasu',
    scripts=scripts,
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
