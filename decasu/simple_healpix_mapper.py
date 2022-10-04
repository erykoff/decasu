import numpy as np
import hpgeom as hpg
import fitsio
import esutil
import healsparse
import time


class SimpleHealpixMapper(object):
    """
    Do a simple map

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    """
    def __init__(self, config):
        self.config = config

    def __call__(self, infile, outfile, band):
        """
        Run the map

        Parameters
        ----------
        infile : `str`
           Input fits file
        outfile : `str`
           Output map file
        """
        # Load in the WCSs and ccd centers
        t = time.time()
        print('Loading WCSs...')
        self._load_wcs_and_centers(infile, band)
        print('... took %.3f seconds.' % (time.time() - t))

        # Render all the boxes
        t = time.time()
        print('Rendering nexp map...')
        nexp_map = self._make_nexp_map()
        print('... took %.3f seconds.' % (time.time() - t))

        # And save
        nexp_map.write(outfile)

    def _load_wcs_and_centers(self, infile, band):
        """
        Load WCSs and compute ccd centers.

        Parameters
        ----------
        infile : `str`
           Input file
        """
        table_in = fitsio.read(infile, ext=1, lower=True, trim_strings=True)

        if 'ctype1' not in table_in.dtype.names:
            dtype = table_in.dtype.descr
            dtype.extend([('ctype1', 'U10'), ('ctype2', 'U10')])

            table = np.zeros(table_in.size, dtype=dtype)
            for name in table_in.dtype.names:
                table[name][:] = np.nan_to_num(table_in[name])
            table['ctype1'][:] = self.config.ctype1
            table['ctype2'][:] = self.config.ctype2
            del table_in
        else:
            table = table_in

        use, = np.where(table[self.config.band_field] == band)
        print('Found %d ccds with %s band' % (use.size, band))
        self.table = table[use]

        self.wcs_list = []
        self.centers = np.zeros(self.table.size, dtype=[('ra_center', 'f8'),
                                                        ('dec_center', 'f8')])
        for i in range(self.table.size):
            wcs = esutil.wcsutil.WCS(self.table[i])
            self.wcs_list.append(wcs)

            ra_center, dec_center = wcs.image2sky(self.table['naxis1'][i]/2.,
                                                  self.table['naxis2'][i]/2.)
            self.centers['ra_center'][i] = ra_center
            self.centers['dec_center'][i] = dec_center

    def _make_nexp_map(self):
        """
        Make the number-of-exposures map

        Returns
        -------
        nexp_map : `healsparse.HealSparseMap`
        """
        # Figure out coverage
        ipnest = hpg.angle_to_pixel(self.config.nside_coverage,
                                    self.centers['ra_center'], self.centers['dec_center'])
        pixels = np.unique(ipnest)

        # Initialize the map and memory
        nexp_map = healsparse.HealSparseMap.make_empty(self.config.nside_coverage,
                                                       self.config.nside,
                                                       np.int32,
                                                       sentinel=0,
                                                       cov_pixels=pixels)

        bad = ((self.table['naxis1'] < self.config.border*2) |
               (self.table['naxis2'] < self.config.border*2))

        # Loop over boxes
        for i, wcs in enumerate(self.wcs_list):
            if bad[i]:
                continue

            x_coords = np.array([self.config.border,
                                 self.config.border,
                                 self.table['naxis1'][i] - self.config.border,
                                 self.table['naxis1'][i] - self.config.border])
            y_coords = np.array([self.config.border,
                                 self.table['naxis2'][i] - self.config.border,
                                 self.table['naxis2'][i] - self.config.border,
                                 self.config.border])
            ra, dec = wcs.image2sky(x_coords, y_coords)
            poly = healsparse.Polygon(ra=ra, dec=dec,
                                      value=1)
            try:
                nexp_map[poly.get_pixels(nside=nexp_map.nside_sparse)] += 1
            except ValueError:
                print('Bad WCS mapping for %d/%d' %
                      (self.table[self.config.exp_field][i],
                       self.table[self.config.ccd_field][i]))

        return nexp_map
