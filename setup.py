from setuptools import setup, find_packages


setup(
    name='decasu',
    packages=find_packages(exclude=('tests')),
    description='DECam Survey Property Maps using healsparse',
    author='Eli Rykoff and others',
    author_email='erykoff@stanford.edu',
    url='https://github.com/erykoff/decasu',
    entry_points={
        'console_scripts': [
            'decasu_hpix_mapper.py = decasu.decasu_hpix_mapper:main',
            'decasu_tile_mapper.py = decasu.decasu_tile_mapper:main',
        ],
    },
    use_scm_version=True,
    setup_requires=['setuptools_scm', 'setuptools_scm_git_archive'],
)
