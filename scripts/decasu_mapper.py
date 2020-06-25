#!/usr/bin/env python

import argparse
import decasu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make survey property maps for DECam')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-i', '--infile', action='store', type=str, required=True,
                        help='Input fits file')
    parser.add_argument('-b', '--band', action='store', type=str, required=True,
                        help='Band to generate map for')
    parser.add_argument('-o', '--outputfile', action='store', type=str, required=True,
                        help='Output map file')
    parser.add_argument('-s', '--simple', action='store_true', required=False,
                        help='Run in simple mode (nexp only)')

    args = parser.parse_args()

    config = decasu.Configuration.load_yaml(args.configfile)

    if args.simple:
        mapper = decasu.SimpleMapper(config)
        mapper(args.infile, args.outputfile, args.band)
    else:
        raise NotImplementedError("Only simple mapper is supported.")
