import numpy as np
import hpgeom as hpg
import sqlite3

from . import decasu_globals

try:
    import lsst.obs.lsst
    from lsst.obs.base import createInitialSkyWcs
    from lsst.afw.cameraGeom import DetectorType
    from lsst.afw.image import VisitInfo, RotType
    import lsst.geom as geom
    lsst_imported = True
except ImportError:
    lsst_imported = False


class LsstWcsDbBuilder:
    """
    Build a WCS table from an LSST database and get intersecting pixels.

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
    def __init__(self, config, dbfile, bands, compute_pixels=True):
        if not lsst_imported:
            raise RuntimeError("Cannot use LsstWcsDbBuilder without Rubin Science Pipelines.")

        self.config = config
        self.compute_pixels = compute_pixels

        con = sqlite3.connect(dbfile)
        cur = con.cursor()

        query_string = ("SELECT observationID, fieldRA, fieldDec, visitExposureTime, airmass, filter, "
                        "fiveSigmaDepth, rotSkyPos, seeingFwhmEff, skyBrightness, "
                        "observationStartMJD, observationStartLST FROM observations ")

        # Just have a basic selection to make the stringing easier.
        where_string = "WHERE observationID>0"

        if len(self.config.lsst_db_additional_selection) > 0:
            where_string = where_string + " and " + self.config.lsst_db_additional_selection

        if len(bands) > 0:
            where_string = where_string + " and filter in (" + ",".join([f"'{band}'" for band in bands]) + ")"

        where_string = where_string + f" and observationStartMJD >= {self.config.mjd_min}"
        where_string = where_string + f" and observationStartMJD <= {self.config.mjd_max}"

        query_string = query_string + where_string + ";"

        res = cur.execute(query_string)
        rows = res.fetchall()

        obs_table = np.array(rows, dtype=[('observationID', 'i8'),
                                          ('fieldRA', 'f8'),
                                          ('fieldDec', 'f8'),
                                          ('visitExposureTime', 'f4'),
                                          ('airmass', 'f4'),
                                          ('filter', 'U10'),
                                          ('fiveSigmaDepth', 'f4'),
                                          ('rotSkyPos', 'f4'),
                                          ('seeingFwhmEff', 'f4'),
                                          ('skyBrightness', 'f4'),
                                          ('observationStartMJD', 'f8'),
                                          ('observationStartLST', 'f8')])

        print(f'Found {len(obs_table)} observations for {len(bands)} bands.')

        if len(bands) == 0:
            # Use them all, record the bands here.
            self.bands = np.unique(obs_table['filter'])
        else:
            self.bands = bands

        instrument = lsst.obs.lsst.LsstCam()
        camera = instrument.getCamera()

        # How many science detectors?
        ids = []
        xsizes = []
        ysizes = []
        for detector in camera:
            if detector.getType() != DetectorType.SCIENCE:
                continue
            ids.append(detector.getId())
            xsizes.append(detector.getBBox().getWidth())
            ysizes.append(detector.getBBox().getHeight())
        ndet = len(ids)
        ids = np.array(ids)
        xsizes = np.array(xsizes)
        ysizes = np.array(ysizes)

        table = np.zeros(obs_table.size*ndet, dtype=[('observationID', 'i8'),
                                                     ('fieldRA', 'f8'),
                                                     ('fieldDec', 'f8'),
                                                     ('detector', 'i4'),
                                                     ('naxis1', 'i4'),
                                                     ('naxis2', 'i4'),
                                                     ('exptime', 'f4'),
                                                     ('airmass', 'f4'),
                                                     ('filter', 'U10'),
                                                     ('fiveSigmaDepth', 'f4'),
                                                     ('rotSkyPos', 'f4'),
                                                     ('seeingFwhmEff', 'f4'),
                                                     ('skyBrightness', 'f4'),
                                                     ('skyvar', 'f4'),
                                                     ('mjd', 'f8'),
                                                     ('mag_zero', 'f4'),
                                                     ('decasu_lst', 'f8')])
        table['observationID'] = np.repeat(obs_table['observationID'], ndet)
        table['fieldRA'] = np.repeat(obs_table['fieldRA'], ndet)
        table['fieldDec'] = np.repeat(obs_table['fieldDec'], ndet)
        table['detector'] = np.tile(ids, obs_table.size)
        table['naxis1'] = np.tile(xsizes, obs_table.size)
        table['naxis2'] = np.tile(ysizes, obs_table.size)
        table['exptime'] = np.repeat(obs_table['visitExposureTime'], ndet)
        table['airmass'] = np.repeat(obs_table['airmass'], ndet)
        table['filter'] = np.repeat(obs_table['filter'], ndet)
        table['fiveSigmaDepth'] = np.repeat(obs_table['fiveSigmaDepth'], ndet)
        table['rotSkyPos'] = np.repeat(obs_table['rotSkyPos'], ndet)
        table['seeingFwhmEff'] = np.repeat(obs_table['seeingFwhmEff'], ndet)
        table['skyBrightness'] = np.repeat(obs_table['skyBrightness'], ndet)
        table['mjd'] = np.repeat(obs_table['observationStartMJD'], ndet)
        table['decasu_lst'] = np.repeat(obs_table['observationStartLST'], ndet)

        # Convert the database columns to decasu values.
        eff_area = 4.*np.pi*(obs_table['seeingFwhmEff']/2.355)**2.
        inv_sqrt_wt = 10.**((obs_table['fiveSigmaDepth'] -
                             self.config.zp_global +
                             2.5*np.log10(5.*np.sqrt(eff_area)))/(-2.5))
        wt = 1./inv_sqrt_wt**2.

        zp = self.config.zp_global - (2.5/2.0)*np.log10(1./wt)

        table['mag_zero'] = np.repeat(zp, ndet)
        table['skyvar'] = 1.0

        print(f'Using {len(table)} detectors for {len(bands)} bands.')

        decasu_globals.table = table
        decasu_globals.lsst_camera = camera

    def __call__(self, row):
        """
        Compute the WCS and intersecting pixels for one row.

        Parameters
        ----------
        row : `int`
            Row to compute WCS and intersecting pixels.

        Returns
        -------
        wcs : `lsst.afw.skyWcs`
        pixels : `list`
            List of nside = `config.nside_run` intersecting pixels.
            Returned if compute_pixels is True in initialization.
        centers : `tuple` [`float`]]
        """
        if (row % 10000) == 0:
            print("Working on WCS index %d" % (row))

        # Link to global table.
        self.table = decasu_globals.table
        # self.camera = decasu_globals.lsst_camera

        detector = decasu_globals.lsst_camera[self.table['detector'][row]]

        boresight = geom.SpherePoint(self.table['fieldRA'][row]*geom.degrees,
                                     self.table['fieldDec'][row]*geom.degrees)
        orientation = self.table['rotSkyPos'][row]*geom.degrees

        visitInfo = VisitInfo(boresightRaDec=boresight,
                              boresightRotAngle=orientation,
                              rotType=RotType.SKY)

        wcs = createInitialSkyWcs(visitInfo, detector, False)

        ra_co, dec_co = wcs.pixelToSkyArray(np.array([0.0, 0.0,
                                                      self.table['naxis1'][row],
                                                      self.table['naxis1'][row]]),
                                            np.array([0.0, self.table['naxis2'][row],
                                                      self.table['naxis2'][row], 0.0]),
                                            degrees=True)
        center = wcs.pixelToSkyArray(np.array([self.table['naxis1'][row]/2.]),
                                     np.array([self.table['naxis2'][row]/2.]),
                                     degrees=True)
        center = [center[0][0], center[1][0]]

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
