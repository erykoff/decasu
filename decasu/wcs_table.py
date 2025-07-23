import numpy as np
import hpgeom as hpg
import fitsio
import esutil

import astropy.units as units
from astropy.time import Time
from astropy.coordinates import EarthLocation

from . import decasu_globals


class WcsTableBuilder:
    """
    Build a WCS table and get intersecting pixels

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    infiles : `list` [`str`]
       List of input files with wcs information
    bands : `list`
       Bands to run.  Empty list means use all.
    compute_pixels : `bool`, optional
       Compute pixels when rendering WCS?
    """
    def __init__(self, config, infiles, bands, pfw_attempt_ids=[], compute_pixels=True):
        self.config = config
        self.compute_pixels = compute_pixels

        fulltable = None

        for infile in infiles:
            table_in = fitsio.read(infile, ext=1, lower=True, trim_strings=True)

            if len(pfw_attempt_ids) > 0:
                a, b = esutil.numpy_util.match(pfw_attempt_ids, table_in['pfw_attempt_id'])
                table_in = table_in[b]

            # Add any extra fields...
            dtype = table_in.dtype.descr
            added_fields = []
            for extra_field in self.config.extra_fields:
                if extra_field not in table_in.dtype.names:
                    dtype.extend([(extra_field, 'U%d' % (len(self.config.extra_fields[extra_field]) + 1))])
                    added_fields.append(extra_field)

            # And add in the hour angle and parallactic angle fields
            dtype.extend([('decasu_lst', 'f8')])

            table = np.zeros(table_in.size, dtype=dtype)
            for name in table_in.dtype.names:
                table[name][:] = np.nan_to_num(table_in[name])
            for field in added_fields:
                table[field][:] = self.config.extra_fields[field]

            if len(config.band_replacement) > 0:
                # Replace bands as configured.
                for b in config.band_replacement:
                    test, = np.where(table[self.config.band_field] == b)
                    table[self.config.band_field][test] = config.band_replacement[b]

            if len(bands) == 0:
                # Use them all, record the bands here
                self.bands = np.unique(table[self.config.band_field])
            else:
                self.bands = bands

                use = None
                for b in bands:
                    if use is None:
                        use = (table[self.config.band_field] == b)
                    else:
                        use |= (table[self.config.band_field] == b)
                table = table[use]

            # Cut to the MJD range.
            use = ((table[self.config.mjd_field] >= self.config.mjd_min) &
                   (table[self.config.mjd_field] <= self.config.mjd_max))
            table = table[use]

            if self.config.zp_sign_swap:
                table[self.config.magzp_field] *= -1.0

            if fulltable is None:
                fulltable = table
            else:
                fulltable = np.append(fulltable, table)

        print('Found %d CCDs for %d bands.' % (len(fulltable), len(bands)))

        print('Computing local sidereal time...')
        loc = EarthLocation(lat=config.latitude*units.degree,
                            lon=config.longitude*units.degree,
                            height=config.elevation*units.m)

        t = Time(fulltable[config.mjd_field], format='mjd', location=loc)
        lst = t.sidereal_time('apparent')
        fulltable['decasu_lst'] = lst.to_value(units.degree)
        print('...done.')

        decasu_globals.table = fulltable

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
           List of nside = `config.nside_run` intersecting pixels.
           Returned if compute_pixels is True in initialization.
        center : `tuple` [`float`]
        """
        if (row % 10000) == 0:
            print("Working on WCS index %d" % (row))
        # Link to global table
        self.table = decasu_globals.table

        if self.config.use_wcs:
            wcs = esutil.wcsutil.WCS(self.table[row])

            ra_co, dec_co = wcs.image2sky(np.array([0.0, 0.0,
                                                    self.table['naxis1'][row], self.table['naxis1'][row]]),
                                          np.array([0.0, self.table['naxis2'][row],
                                                    self.table['naxis2'][row], 0.0]))
            center = wcs.image2sky([self.table['naxis1'][row]/2.],
                                   [self.table['naxis2'][row]/2.])
        else:
            # Don't set wcs to None because that has meaning that this
            # is a bad ccd
            wcs = 0

            ra_co = np.array([self.table[field][row] for field in self.config.ra_corner_fields])
            dec_co = np.array([self.table[field][row] for field in self.config.dec_corner_fields])
            center = [np.mean(ra_co), np.mean(dec_co)]

        if self.compute_pixels:
            try:
                pixels = hpg.query_polygon(self.config.nside_run, ra_co, dec_co, inclusive=True, fact=16)
            except RuntimeError:
                # Bad WCS
                pixels = np.array([], dtype=np.int64)
                wcs = None

            return wcs, pixels, center
        else:
            return wcs, center
