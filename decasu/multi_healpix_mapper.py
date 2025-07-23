import os
import numpy as np
import multiprocessing
import time
import esutil
import glob

from .wcs_table import WcsTableBuilder
from .lsst_wcs_db import LsstWcsDbBuilder
from .lsst_wcs_consdb import LsstWcsConsDbBuilder
from .region_mapper import RegionMapper
from .healpix_consolidator import HealpixConsolidator
from .utils import op_str_to_code
from . import decasu_globals


class MultiHealpixMapper:
    """
    Map a combination of bands/pixels

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

    def __call__(self, infile, bands=[], pixels=[], clear_intermediate_files=True):
        """
        Compute maps for a combination of bands/pixels

        Parameters
        ----------
        infile : `str`
           Name of input file with wcs and map information
        bands : `list`, optional
           List of bands to run.  If blank, run all.
        pixels : `list`, optional
           List of pixels to run (nside=`config.nside_run`).
           If blank, run all.
        clear_intermediate_files : `bool`, optional
           Clear intermediate files when done?
        """
        # First build the wcs's
        print('Reading input table...')
        if self.config.use_lsst_db:
            wcs_builder = LsstWcsDbBuilder(self.config, infile, bands)
        elif self.config.use_lsst_consdb:
            wcs_builder = LsstWcsConsDbBuilder(self.config, infile, bands)
        else:
            wcs_builder = WcsTableBuilder(self.config, [infile], bands)

        if len(decasu_globals.table) == 0:
            print('No observations in specified ranges.')
            return

        print('Generating WCSs...')
        t = time.time()
        mp_ctx = multiprocessing.get_context("fork")
        pool = mp_ctx.Pool(processes=self.ncores)
        wcs_list, pixel_list, center_list = zip(*pool.map(wcs_builder,
                                                          range(len(decasu_globals.table)),
                                                          chunksize=1))
        pool.close()
        pool.join()
        print('Time elapsed: ', time.time() - t)

        # Copy into/out of globals
        decasu_globals.wcs_list = wcs_list
        table = decasu_globals.table

        # Split into pixels
        pixel_arr = np.concatenate(pixel_list)
        pixel_indices = np.zeros(len(wcs_list) + 1, dtype=np.int32)

        ctr = 0
        for i in range(len(wcs_list)):
            pixel_indices[i] = ctr
            ctr += len(pixel_list[i])
        pixel_indices[-1] = ctr

        h, rev = esutil.stat.histogram(pixel_arr, rev=True)
        u, = np.where(h > 0)
        runpix_list = []
        wcsindex_list = []
        for ind in u:
            i1a = rev[rev[ind]: rev[ind + 1]]
            wcs_inds = np.searchsorted(pixel_indices, i1a, side='left')
            miss, = np.where(pixel_indices[wcs_inds] != i1a)
            wcs_inds[miss] -= 1

            # This could probably be more efficient
            for band in wcs_builder.bands:
                ok, = np.where(table[self.config.band_field][wcs_inds] == band)
                if ok.size > 0:
                    runpix_list.append(pixel_arr[i1a[0]])
                    wcsindex_list.append(wcs_inds[ok])

        # Generate maps
        region_mapper = RegionMapper(self.config, self.outputpath, 'pixel',
                                     self.config.nside_coverage)

        values = zip(runpix_list, wcsindex_list)

        print('Generating maps for %d pixels...' % (len(runpix_list)))
        t = time.time()
        mp_ctx = multiprocessing.get_context("fork")
        pool = mp_ctx.Pool(processes=self.ncores)
        pool.starmap(region_mapper, values, chunksize=1)
        pool.close()
        pool.join()
        print('Time elapsed: ', time.time() - t)

        # Consolidate
        print('Consolidating maps...')
        fname_list = []
        mapfiles_list = []
        for map_type in self.config.map_types.keys():
            for operation in self.config.map_types[map_type]:
                op_code = op_str_to_code(operation)
                for band in wcs_builder.bands:
                    # Get full map filename
                    fname = os.path.join(self.outputpath, self.config.map_filename(band,
                                                                                   map_type,
                                                                                   op_code))
                    # Check if file exists
                    if os.path.isfile(fname):
                        continue

                    # Assemble all the individual pixel maps
                    fname_template = self.config.healpix_map_filename_template(band,
                                                                               map_type,
                                                                               op_code)

                    mapfiles = sorted(glob.glob(os.path.join(self.outputpath,
                                                             '%d_?????' % (self.config.nside_run),
                                                             fname_template)))
                    fname_list.append(fname)
                    mapfiles_list.append(mapfiles)

        hpix_consolidator = HealpixConsolidator(self.config, clear_intermediate_files)

        values = zip(fname_list, mapfiles_list)

        t = time.time()
        mp_ctx = multiprocessing.get_context("fork")
        pool = mp_ctx.Pool(processes=self.ncores)
        pool.starmap(hpix_consolidator, values, chunksize=1)
        pool.close()
        pool.join()
        print('Time elapsed: ', time.time() - t)

        # Clean up
        if clear_intermediate_files:
            print('Cleaning intermediate directories...')
            directories = sorted(glob.glob(os.path.join(self.outputpath,
                                                        '%d_?????' % (self.config.nside_run))))
            for d in directories:
                if not os.listdir(d):
                    # Clear directory, it is empty
                    os.rmdir(d)
