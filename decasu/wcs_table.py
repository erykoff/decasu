import numpy as np
import healpy as hp
import fitsio
import esutil

from . import decasu_globals


class WcsTableBuilder(object):
    """
    Build a WCS table and get intersecting pixels

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    infile : `str`
       Name of input file with wcs information
    bands : `list`
       Bands to run.  Empty list means use all.
    """
    def __init__(self, config, infile, bands):
        self.config = config

        table_in = fitsio.read(infile, ext=1, lower=True, trim_strings=True)

        # Add any extra fields...
        dtype = table_in.dtype.descr
        added_fields = []
        for extra_field in self.config.extra_fields:
            if extra_field not in table_in.dtype.names:
                dtype.extend([(extra_field, 'U%d' % (len(self.config.extra_fields[extra_field]) + 1))])
                added_fields.append(extra_field)

        if len(added_fields) == 0:
            table = table_in
        else:
            table = np.zeros(table_in.size, dtype=dtype)
            for name in table_in.dtype.names:
                table[name][:] = np.nan_to_num(table_in[name])
            for field in added_fields:
                table[field][:] = self.config.extra_fields[field]

        if len(bands) == 0:
            # Use them all, record the bands here
            self.bands = np.unique(table['band'])
        else:
            self.bands = bands

            use = None
            for b in bands:
                if use is None:
                    use = (table['band'] == b)
                else:
                    use |= (table['band'] == b)
            table = table[use]

        if self.config.zp_sign_swap:
            table[self.config.magzp_field] *= -1.0

        print('Found %d CCDs for %d bands.' % (len(table), len(bands)))

        decasu_globals.table = table
        self.nrows = len(table)

    def __call__(self, row):
        """
        Compute the WCS and intersecting pixels for one row.

        Parameters
        ----------
        row : `int`
           Row to compute WCS and intersecting pixels

        Returns
        -------
        wcs : `esutil.wcsutil.WCS`
        pixels : `list`
           List of nside = `config.nside_run` intersecting pixels
        """
        if (row % 10000) == 0:
            print("Working on WCS index %d" % (row))
        # Link to global table
        self.table = decasu_globals.table

        wcs = esutil.wcsutil.WCS(self.table[row])

        ra_co, dec_co = wcs.image2sky(np.array([0.0, 0.0,
                                                self.table['naxis1'][row], self.table['naxis1'][row]]),
                                      np.array([0.0, self.table['naxis2'][row],
                                                self.table['naxis2'][row], 0.0]))
        center = wcs.image2sky([self.table['naxis1'][row]/2.],
                               [self.table['naxis2'][row]/2.])
        vertices = hp.ang2vec(ra_co, dec_co, lonlat=True)
        try:
            pixels = hp.query_polygon(self.config.nside_run, vertices, nest=True, inclusive=True, fact=16)
        except RuntimeError:
            # Bad WCS
            pixels = np.array([], dtype=np.int64)
            wcs = None

        return wcs, pixels, center
