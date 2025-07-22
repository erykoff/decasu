import numpy as np
import hpgeom as hpg

from astropy.table import Table
import astropy.units as units
from astropy.time import Time
from astropy.coordinates import EarthLocation

from . import decasu_globals

try:
    import lsst.obs.lsst
    import lsst.sphgeom
    import psycopg
    lsst_imported = True
except ImportError:
    lsst_imported = False


class LsstWcsConsDbBuilder:
    """
    Build a WCS table from the LSST Consolidated Database and get intersecting
    pixels.

    Parameters
    ----------
    config : `Configuration`
        decasu configuration object.
    dbfile : `str`
        Input database file.
    bands : `list`
        Bands to run.  Empty list means use all.
    compute_pixels : `bool`, optional
        Compute pixels when rendering WCS?
    """
    def __init__(self, config, dbstring, bands, compute_pixels=True):
        if not lsst_imported:
            raise RuntimeError("Cannot use LsstWcsConsDbBuilder without Rubin Science Pipelines.")

        self.config = config
        self.compute_pixels = compute_pixels

        query_string = (
            "SELECT cvq.eff_time, cvq.psf_sigma, "
            "cvq.sky_bg, cvq.sky_noise, cvq.zero_point, "
            "cv.detector, cv.visit_id, cv.s_region, "
            "v.band, v.exp_time, v.exp_midpt_mjd, v.sky_rotation "
            "FROM cdb_LSSTCam.ccdvisit1_quicklook as cvq, cdb_LSSTCam.ccdvisit1 as cv, "
            "cdb_LSSTCam.visit1 as v "
        )
        where_string = (
            "WHERE cvq.ccdvisit_id=cv.ccdvisit_id and "
            "cv.visit_id=v.visit_id and "
            "detector<189 and cvq.zero_point is not null "
        )

        if len(self.config.lsst_db_additional_selection) > 0:
            where_string = where_string + " and " + self.config.lsst_db_additional_selection

        if len(bands) > 0:
            where_string = where_string + " and v.band in (" + ",".join([f"'{band}'" for band in bands]) + ")"

        where_string = where_string + f" and v.exp_midpt_mjd >= {self.config.mjd_min}"
        where_string = where_string + f" and v.exp_midpt_mjd <= {self.config.mjd_max}"

        query_string = query_string + where_string + ";"

        with psycopg.Connection.connect(dbstring) as conn:
            cur = conn.execute(query_string)
            rows = cur.fetchall()

        db_table = Table(
            np.asarray(
                rows,
                dtype=[
                    ("eff_time", "f4"),
                    ("psf_sigma", "f4"),
                    ("sky_bg", "f4"),
                    ("sky_noise", "f4"),
                    ("zero_point", "f4"),
                    ("detector", "i4"),
                    ("visit_id", "i8"),
                    ("s_region", "U200"),
                    ("band", "U2"),
                    ("exptime", "f4"),
                    ("mjd", "f8"),
                    ("sky_rotation", "f4"),
                ],
            ),
        )

        if len(bands) == 0:
            self.bands = np.unique(db_table["band"])
        else:
            self.bands = bands

        print(f"Found {len(db_table)} detector visits for {len(self.bands)} bands.")

        # Add extra columns.
        db_table["ra_center"] = np.zeros(len(db_table))
        db_table["dec_center"] = np.zeros(len(db_table))
        db_table["decasu_lst"] = np.zeros(len(db_table))
        db_table["skyvar"] = db_table["sky_noise"]**2.

        print("Computing local sidereal time...")
        loc = EarthLocation(lat=config.latitude*units.degree,
                            lon=config.longitude*units.degree,
                            height=config.elevation*units.m)

        t = Time(db_table[config.mjd_field], format="mjd", location=loc)
        lst = t.sidereal_time("apparent")
        db_table["decasu_lst"] = lst.to_value(units.degree)

        instrument = lsst.obs.lsst.LsstCam()
        camera = instrument.getCamera()

        decasu_globals.table = db_table
        decasu_globals.lsst_camera = camera

    def __call__(self, row):
        """
        Compute intersecting pixels for onw row.

        Parameters
        ----------
        row : `int`
            Row to compute intersecting pixels.

        Returns
        -------
        wcs : `int`
            Placeholder.
        pixels : `list`
            List of nside = `config.nside_run` intersecting pixels.
            Returned if compute_pixels is True in initialization.
        centers : `tuple` [`float`]]
        """
        if (row % 10000) == 0:
            print("Working on WCS index %d" % (row))

        # Link to global table.
        self.table = decasu_globals.table

        region_str = self.table["s_region"][row]

        region = lsst.sphgeom.Region.from_ivoa_pos("".join(region_str.split("ICRS")).upper())
        centroid = lsst.sphgeom.LonLat(region.getCentroid())
        center = [centroid.getLon().asDegrees(), centroid.getLat().asDegrees()]

        self.table["ra_center"][row] = center[0]
        self.table["dec_center"][row] = center[1]

        if self.compute_pixels:
            vertices = np.asarray([[v.x(), v.y(), v.z()] for v in region.getVertices()])
            pixels = hpg.query_polygon_vec(self.config.nside_run, vertices, inclusive=True, fact=16)
            return 0, pixels, center
        else:
            return 0, center
