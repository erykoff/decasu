import os
import numpy as np
from multiprocessing import Pool
import time
import esutil
import fitsio
import healpy as hp
import healsparse

from .wcs_table import WcsTableBuilder
from .region_mapper import RegionMapper
from .utils import op_str_to_code, read_maskfiles
from . import decasu_globals


class MultiTileMapper(object):
    """
    Map a combination of tiles for a single band.

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    outputpath : `str`
       Base path for output files
    ncores : `int`
       Number of cores to run with
    """
    def __init__(self, config, outputpath, ncores=1):
        self.config = config
        self.outputpath = outputpath
        self.ncores = ncores

        if not os.path.isdir(outputpath):
            raise RuntimeError("Outputpath %s does not exist." % (outputpath))

    def __call__(self, coaddtilefile, imagefiles, band,
                 coaddtiles=[], bleedtrailfiles=[], streakfiles=[],
                 starfiles=[], clear_intermediate_files=True,
                 clobber=False):
        """
        Compute maps for a combination of tiles

        Parameters
        ----------
        coaddtilefile : `str`
           Name of file with coadd tile geometry
        imagefiles : `list` [`str`]
           List of image data files with wcs and quantities
        band : `str`
           Name of band to run
        coaddtiles : `list` [`str`], optional
           List of coadd tilenames to run on.  Default is all in coaddtilefile.
        bleedtrailfiles : `list` [`str`], optional
           List of bleed trail region files.
        streakfiles : `list` [`str`], optional
           List of streak region files.
        starfiles : `list` [`str`], optional
           list of saturated star region files.
        clobber : `bool`, optional
           Clobber any existing files
        clear_intermediate_files : `bool`, optional
           Clear intermediate files when done?
        """
        # Get the coadd tile information
        decasu_globals.tile_info = fitsio.read(coaddtilefile, ext=1, lower=True, trim_strings=True)
        use, = np.where(decasu_globals.tile_info['band'] == band)
        decasu_globals.tile_info = decasu_globals.tile_info[use]

        # Confirm that we have a unique list of tiles
        utiles = np.unique(decasu_globals.tile_info['tilename'])
        if utiles.size != decasu_globals.tile_info.size:
            raise RuntimeError("Input %s does not have unique tiles for band %s" %
                               (coaddtilefile, band))

        pfw_attempt_ids = []
        if len(coaddtiles) > 0:
            utiles = np.unique(coaddtiles)
            if utiles.size != len(coaddtiles):
                raise RuntimeError("Input list of coaddtiles is not unique.")

            tile_info_names = list(decasu_globals.tile_info['tilename'])
            for coaddtile in coaddtiles:
                try:
                    pid = decasu_globals.tile_info['pfw_attempt_id'][tile_info_names.index(coaddtile)]
                    pfw_attempt_ids.append(pid)
                except ValueError:
                    print("Warning: tile %s not in coadd tile info table." % (coaddtile))
                    continue
        else:
            # Use all of them that are in the tile_info file.
            pfw_attempt_ids = np.unique(decasu_globals.tile_info['pfw_attempt_id'])

        # Build the WCSs
        print('Reading input table(s)...')
        wcs_builder = WcsTableBuilder(self.config, imagefiles, [band],
                                      pfw_attempt_ids=pfw_attempt_ids,
                                      compute_pixels=False)

        # Find the unique tiles in the decasu_globals.table
        # Check if any of these have the complete set of output files.
        # If so, remove them from decasu_globals.table
        if not clobber:
            print("Checking for completed tiles...")
            tilenames, table_indices = np.unique(decasu_globals.table['tilename'],
                                                 return_index=True)
            complete_tile_indices = []
            for i, tilename in enumerate(tilenames):
                tilepath = os.path.join(self.outputpath,
                                        self.config.tile_relpath(tilename))
                if not os.path.isdir(tilepath):
                    # Nothing to see here, move along
                    continue
                input_map_filename = os.path.join(tilepath,
                                                  self.config.tile_input_filename(band,
                                                                                  tilename))
                if not os.path.isfile(input_map_filename):
                    continue

                found = False
                for map_type in self.config.map_types.keys():
                    if found:
                        continue
                    for j, operation in enumerate(self.config.map_types[map_type]):
                        if found:
                            continue
                        op_code = op_str_to_code(operation)
                        fname = os.path.join(tilepath,
                                             self.config.tile_map_filename(band,
                                                                           tilename,
                                                                           map_type,
                                                                           op_code))
                        if not os.path.isfile(fname):
                            found = True
                            continue

                if not found:
                    complete_tile_indices.append(i)

            # Now we need to remove all of these from the list
            if len(complete_tile_indices) > 0:
                print('Not computing %d tiles already completed of %d.' %
                      (len(complete_tile_indices), len(utiles)))
                complete_tile_indices = np.array(complete_tile_indices)
                comp_ids = decasu_globals.table['pfw_attempt_id'][table_indices[complete_tile_indices]]
                aa, bb = esutil.numpy_util.match(comp_ids,
                                                 decasu_globals.table['pfw_attempt_id'])
                decasu_globals.table = np.delete(decasu_globals.table, bb)

        if len(decasu_globals.table) == 0:
            print('No tiles to run.  Skipping to consolidation...')
        else:
            print('Generating WCSs...')
            t = time.time()
            pool = Pool(processes=self.ncores)
            wcs_list, center_list = zip(*pool.map(wcs_builder, range(len(decasu_globals.table)),
                                                  chunksize=1))
            pool.close()
            pool.join()
            print('Time elapsed: ', time.time() - t)

            # Copy into/out of globals
            decasu_globals.wcs_list = wcs_list
            table = decasu_globals.table

            # Read in tables
            expnums = np.unique(table['expnum'])
            if len(bleedtrailfiles) > 0:
                print('Reading bleed trail table...')
                decasu_globals.bleed_table = read_maskfiles(expnums,
                                                            bleedtrailfiles)
            if len(streakfiles) > 0:
                print('Reading streak table...')
                decasu_globals.streak_table = read_maskfiles(expnums,
                                                             streakfiles)
            if len(starfiles) > 0:
                print('Reading saturated star table...')
                decasu_globals.satstar_table = read_maskfiles(expnums,
                                                              starfiles)
                gd, = np.where(decasu_globals.satstar_table['radius'] < self.config.satstar_maxrad)
                decasu_globals.satstar_table = decasu_globals.satstar_table[gd]

            # Split into tiles (by pfw_attempt_id)
            h, rev = esutil.stat.histogram(table['pfw_attempt_id'], rev=True)
            u, = np.where(h > 0)
            runtile_list = []
            wcsindex_list = []
            for ind in u:
                i1a = rev[rev[ind]: rev[ind + 1]]
                runtile_list.append(table['tilename'][i1a[0]])
                wcsindex_list.append(i1a)

            nside_coverage_tile = self._compute_nside_coverage_tile(decasu_globals.tile_info[0])

            # Generate maps
            region_mapper = RegionMapper(self.config, self.outputpath, 'tile',
                                         nside_coverage_tile)

            values = zip(runtile_list, wcsindex_list, [clobber]*len(runtile_list))

            print('Generating maps for %d tiles...' % (len(runtile_list)))
            t = time.time()
            pool = Pool(processes=self.ncores)
            pool.starmap(region_mapper, values, chunksize=1)
            pool.close()
            pool.join()
            print('Time elapsed: ', time.time() - t)

        # Consolidate
        print('Finding maps to consolidate from %d tiles.' % (len(utiles)))

        cov_pixels = None
        for map_type in self.config.map_types.keys():
            # Skip the "coverage" map which can't be easily combined.
            if map_type == 'coverage':
                continue
            for operation in self.config.map_types[map_type]:
                op_code = op_str_to_code(operation)

                output_fname = os.path.join(self.outputpath,
                                            self.config.map_filename(band,
                                                                     map_type,
                                                                     op_code))
                if os.path.isfile(output_fname) and not clobber:
                    print("Found %s.  Skipping..." % (output_fname))
                    continue

                start_time = time.time()
                input_files = []
                for tilename in utiles:
                    tilepath = os.path.join(self.outputpath,
                                            self.config.tile_relpath(tilename))
                    fname = os.path.join(tilepath,
                                         self.config.tile_map_filename(band,
                                                                       tilename,
                                                                       map_type,
                                                                       op_code))
                    if os.path.isfile(fname):
                        input_files.append(fname)

                print("Found %d input files for %s in %.2f s" % (len(input_files),
                                                                 os.path.basename(output_fname),
                                                                 time.time() - start_time))
                start_time = time.time()
                # We can get the coverage pixels just once for all the maps
                if cov_pixels is None:
                    cov_pixels = self._get_coverage_pixels(input_files)
                self._simple_consolidator(input_files, output_fname, cov_pixels, clobber=clobber)
                print("...done consolidating %s in %.2f s" % (os.path.basename(output_fname),
                                                              time.time() - start_time))

        # Clean up
        if clear_intermediate_files:
            print('Not cleaning up yet...')

    def _compute_nside_coverage_tile(self, row):
        """
        Compute the optimal coverage nside for a tile.

        Parameters
        ----------
        row : `np.ndarray`
           Row of the tile geometry table

        Returns
        -------
        nside_coverage_tile : `int`
        """
        if row['crossra0'] == 'Y':
            delta_ra = (row['uramax'] - (row['uramin'] - 360.0))
        else:
            delta_ra = row['uramax'] - row['uramin']
        delta_dec = row['udecmax'] - row['udecmin']

        tile_area = delta_ra*delta_dec*np.cos(np.deg2rad((row['udecmin'] + row['udecmax'])/2.))
        nside_coverage_tile = 32
        while hp.nside2pixarea(nside_coverage_tile, degrees=True) > tile_area:
            nside_coverage_tile = int(2*nside_coverage_tile)
        nside_coverage_tile = int(nside_coverage_tile / 2)

        return nside_coverage_tile

    def _get_coverage_pixels(self, input_files):
        """
        Get the coverage pixels from the input files

        Parameters
        ----------
        input_files : `list` of `str`

        Returns
        -------
        cov_pixels : `np.ndarray`
        """
        cov_map = healsparse.HealSparseCoverage.make_empty(self.config.nside_coverage,
                                                           self.config.nside)
        cov_mask = cov_map.coverage_mask
        for f in input_files:
            cov_map_temp = healsparse.HealSparseCoverage.read(f)
            gd = (cov_map_temp.coverage_mask)
            cov_mask[gd] = True

        cov_pixels, = np.where(cov_mask)

        return cov_pixels

    def _simple_consolidator(self, input_files, output_fname, cov_pixels, clobber=False):
        """
        Do a simple in-memory consolidation.

        Parameters
        ----------
        input_files : `list` of `str`
        output_fname : `str`
        clobber : `bool`, optional
        cov_pixels : `np.ndarray`
           Coverage pixels
        """
        if os.path.isfile(output_fname) and not clobber:
            # Nothing to do here.
            return

        combined_map = None

        for f in input_files:
            map_temp = healsparse.HealSparseMap.read(f)

            if combined_map is None:
                combined_map = healsparse.HealSparseMap.make_empty(self.config.nside_coverage,
                                                                   self.config.nside,
                                                                   map_temp.dtype,
                                                                   sentinel=map_temp._sentinel,
                                                                   cov_pixels=cov_pixels)
            valid_pixels = map_temp.valid_pixels
            combined_map[valid_pixels] = map_temp[valid_pixels]

        combined_map.write(output_fname, clobber=clobber)
