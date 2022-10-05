import argparse

from .configuration import Configuration
from .simple_healpix_mapper import SimpleHealpixMapper
from .multi_healpix_mapper import MultiHealpixMapper


def main():
    parser = argparse.ArgumentParser(description='Make survey property maps for DECam using healpix regions')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-i', '--infile', action='store', type=str, required=True,
                        help='Input fits or database file')
    parser.add_argument('-b', '--bands', action='store', type=str, required=False,
                        help='Bands to generate map for, comma delimited')
    parser.add_argument('-n', '--ncores', action='store', type=int, required=False,
                        default=1, help='Number of cores to run on.')
    parser.add_argument('-o', '--outputpath', action='store', type=str, required=True,
                        help='Output path')
    parser.add_argument('-p', '--pixels', action='store', type=str, required=False,
                        help='Pixels to run on, comma delimited')
    parser.add_argument('-s', '--simple', action='store_true', required=False,
                        help='Run in simple mode (nexp only)')
    parser.add_argument('-k', '--keep_intermediate_files', action='store_true',
                        required=False, help='Keep intermediate files')

    args = parser.parse_args()

    config = Configuration.load_yaml(args.configfile)

    if args.bands is None:
        bands = []
    else:
        bands = [b for b in args.bands.split(',')]

    if args.pixels is None:
        pixels = []
    else:
        pixels = [int(p) for p in args.pixels.split(',')]

    if args.simple:
        mapper = SimpleHealpixMapper(config)
        mapper(args.infile, 'blah.hs', bands[0])
    else:
        mapper = MultiHealpixMapper(config, args.outputpath, ncores=args.ncores)
        mapper(args.infile, bands=bands, pixels=pixels,
               clear_intermediate_files=not args.keep_intermediate_files)
