#!/usr/bin/env python

import argparse
import decasu

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Make survey property maps for DECam using coadd tiles')

    parser.add_argument('-c', '--configfile', action='store', type=str, required=True,
                        help='YAML config file')
    parser.add_argument('-i', '--imagefiles', action='store', type=str, required=True,
                        help='Input image files, comma delimited')
    parser.add_argument('-b', '--band', action='store', type=str, required=True,
                        help='Band to generate maps.')
    parser.add_argument('-t', '--coaddtilefile', action='store', type=str, required=True,
                        help='Coadd tile file')
    parser.add_argument('-B', '--bleedtrailfiles', action='store', type=str, required=False,
                        help='Bleed trail region files, comma delimited')
    parser.add_argument('-S', '--streakfiles', action='store', type=str, required=False,
                        help='Streak region files, comma delimited')
    parser.add_argument('-s', '--starfiles', action='store', type=str, required=False,
                        help='Saturated star region files, comma delimited')
    parser.add_argument('-o', '--outputpath', action='store', type=str, required=True,
                        help='Output path')
    parser.add_argument('-T', '--coaddtiles', action='store', type=str, required=False,
                        help='Coadd tiles to run, comma delimited')
    parser.add_argument('-k', '--keep_intermediate_files', action='store_true',
                        required=False, help='Keep intermediate files')
    parser.add_argument('-n', '--ncores', action='store', type=int, required=False,
                        default=1, help='Number of cores to run on.')

    args = parser.parse_args()

    config = decasu.Configuration.load_yaml(args.configfile)

    band = args.band

    imagefiles = args.imagefiles.split(',')

    if args.bleedtrailfiles is None:
        bleedtrailfiles = []
    else:
        bleedtrailfiles = args.bleedtrailfiles.split(',')

    if args.streakfiles is None:
        streakfiles = []
    else:
        streakfiles = args.streakfiles.split(',')

    if args.starfiles is None:
        starfiles = []
    else:
        starfiles = args.starfiles.split(',')

    if args.coaddtiles is None:
        coaddtiles = []
    else:
        coaddtiles = args.coaddtiles.split(',')

    mapper = decasu.MultiTileMapper(config, args.outputpath, ncores=args.ncores)
    mapper(args.coaddtilefile, imagefiles, band, coaddtiles=coaddtiles,
           bleedtrailfiles=bleedtrailfiles, streakfiles=streakfiles,
           starfiles=starfiles, clear_intermediate_files=not args.keep_intermediate_files)
