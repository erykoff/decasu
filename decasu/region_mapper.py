import os
import numpy as np
import healsparse
import healpy as hp

import astropy.units as units
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import coord

from .utils import op_str_to_code
from .utils import OP_NONE, OP_SUM, OP_MEAN, OP_WMEAN, OP_MIN, OP_MAX
from . import decasu_globals


os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'


class RegionMapper(object):
    """
    Map a single region (healpix pixel or tile)

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    outputpath : `str`
       Output path name
    mode : `str`
       Must be "tile" or "pixel"
    nside_coverage_region : `int`
       coverage map nside for region masking
    """
    def __init__(self, config, outputpath, mode, nside_coverage_region):
        self.config = config
        self.outputpath = outputpath
        self.nside_coverage_region = nside_coverage_region

        if mode == 'tile':
            self.tilemode = True
        elif mode == 'pixel':
            self.tilemode = False
        else:
            raise ValueError("mode must be 'tile' or 'pixel'")

    def __call__(self, hpix_or_tilename, indices):
        """
        Run all the configured maps for a given hpix (nside self.config.nside_run).

        Parameters
        ----------
        hpix_or_tilename : `int` or `str
           Healpix or tilename to run
        indices : `np.ndarray`
           Indices of table/wcs_list in the given healpix/tile
        """
        self.table = decasu_globals.table
        self.wcs_list = decasu_globals.wcs_list
        self.tile_info = decasu_globals.tile_info
        self.streak_table = decasu_globals.streak_table
        self.bleed_table = decasu_globals.bleed_table
        self.satstar_table = decasu_globals.satstar_table

        self.band = self.table['band'][indices[0]]

        loc = EarthLocation(lat=self.config.latitude*units.degree,
                            lon=self.config.longitude*units.degree,
                            height=self.config.elevation*units.m)

        if self.tilemode:
            tilename = hpix_or_tilename
            print("Computing maps for tile %s with %d inputs" % (tilename, len(indices)))

            # Create the path for the files if necessary
            os.makedirs(os.path.join(self.outputpath,
                                     self.config.tile_relpath(tilename)),
                        exist_ok=True)

            # Check if this exists and read or build
            input_map_filename = os.path.join(self.outputpath,
                                              self.config.tile_relpath(tilename),
                                              self.config.tile_input_filename(self.band,
                                                                              tilename))
            if os.path.isfile(input_map_filename):
                input_map = healsparse.HealSparseMap.read(input_map_filename)
            else:
                input_map = self.build_region_input_map(indices, tilename=tilename)
                input_map.write(input_map_filename)
        else:
            hpix = hpix_or_tilename
            print("Computing maps for pixel %d with %d inputs" % (hpix, len(indices)))

            # Create the path for the files if necessary
            os.makedirs(os.path.join(self.outputpath,
                                     self.config.healpix_relpath(hpix)),
                        exist_ok=True)

            # Check if this exists and read or build
            input_map_filename = os.path.join(self.outputpath,
                                              self.config.healpix_relpath(hpix),
                                              self.config.healpix_input_filename(self.band,
                                                                                 hpix))
            if os.path.isfile(input_map_filename):
                input_map = healsparse.HealSparseMap.read(input_map_filename)
            else:
                input_map = self.build_region_input_map(indices, hpix=hpix)
                input_map.write(input_map_filename)

        valid_pixels, vpix_ra, vpix_dec = input_map.valid_pixels_pos(lonlat=True,
                                                                     return_pixels=True)
        npixels = len(valid_pixels)

        if npixels == 0:
            return

        pixel_min = valid_pixels.min()

        # Figure out how many maps we need to make and initialize memory
        map_values_list = []
        map_operation_list = []
        map_fname_list = []
        has_zenith_quantity = False

        for map_type in self.config.map_types.keys():
            if map_type == 'airmass' or map_type.startswith('dcr') or map_type == 'parallactic':
                has_zenith_quantity = True

            if map_type in ['nexp']:
                map_dtype = np.int32
            elif map_type in ['coverage']:
                map_dtype = healsparse.WIDE_MASK
            else:
                map_dtype = np.float64

            n_operations = len(self.config.map_types[map_type])
            map_values = np.zeros((npixels, n_operations), dtype=map_dtype)
            op_list = []
            fname_list = []
            for j, operation in enumerate(self.config.map_types[map_type]):
                op_code = op_str_to_code(operation)

                # Check to see if file already exists
                if self.tilemode:
                    fname = os.path.join(self.outputpath,
                                         self.config.tile_relpath(tilename),
                                         self.config.tile_map_filename(self.band,
                                                                       tilename,
                                                                       map_type,
                                                                       op_code))
                else:
                    fname = os.path.join(self.outputpath,
                                         self.config.healpix_relpath(hpix),
                                         self.config.healpix_map_filename(self.band,
                                                                          hpix,
                                                                          map_type,
                                                                          op_code))
                if os.path.isfile(fname):
                    op_code = OP_NONE

                op_list.append(op_code)
                fname_list.append(fname)

                if op_code == OP_MIN or op_code == OP_MAX:
                    # We use fmin and fmax, so nans get overwritten
                    map_values[:, j] = np.nan

            map_values_list.append(map_values)
            map_operation_list.append(op_list)
            map_fname_list.append(fname_list)

        # Check if all the operations are OP_NONE and we can skip entirely
        any_to_compute = False
        any_weights = False
        for i, map_type in enumerate(self.config.map_types.keys()):
            for op in map_operation_list[i]:
                if op != OP_NONE:
                    any_to_compute = True
                if op == OP_WMEAN:
                    any_weights = True

        if not any_to_compute:
            # Everything is already there
            if self.tilemode:
                print("All maps for tile %s are already computed.  Skipping..." % (tilename))
            else:
                print("All maps for pixel %d are already computed.  Skipping..." % (hpix))
            return

        weights = np.zeros(npixels)
        nexp = np.zeros(npixels, dtype=np.int32)

        # Compute the maps
        for i, ind in enumerate(indices):
            bit_a = i*2
            bit_b = i*2 + 1

            use_a, = np.where(input_map.check_bits_pix(valid_pixels, [bit_a]))
            use_b, = np.where(input_map.check_bits_pix(valid_pixels, [bit_b]))

            if use_a.size == 0 and use_b.size == 0:
                # Nothing to see here, move along.
                continue

            if any_weights:
                pixel_weights_a = input_map.metadata['B%04dWT' % (bit_a)]
                pixel_weights_b = input_map.metadata['B%04dWT' % (bit_b)]
            else:
                pixel_weights_a = 1.0
                pixel_weights_b = 1.0

            weights[use_a] += pixel_weights_a
            weights[use_b] += pixel_weights_b

            use = np.concatenate((use_a, use_b))
            pixel_weights = np.concatenate((np.full(use_a.size, pixel_weights_a),
                                            np.full(use_b.size, pixel_weights_b)))
            nexp[use] += 1

            if has_zenith_quantity:
                zenith, par_angle = self._compute_zenith_and_par_angles(loc, self.table['mjd_obs'][ind],
                                                                        np.median(vpix_ra[use]),
                                                                        np.median(vpix_dec[use]))

            for i, map_type in enumerate(self.config.map_types.keys()):
                if map_type == 'nexp':
                    value = 1
                elif map_type == 'maglimit':
                    # We compute this below from the weights
                    value = 0.0
                elif map_type == 'coverage':
                    continue
                elif map_type == 'airmass':
                    value = self._compute_airmass(zenith)
                elif map_type == 'dcr_dra':
                    value = np.tan(zenith)*np.sin(par_angle)
                elif map_type == 'dcr_ddec':
                    value = np.tan(zenith)*np.cos(par_angle)
                elif map_type == 'dcr_e1':
                    value = (np.tan(zenith)**2.)*np.cos(2*par_angle)
                elif map_type == 'dcr_e2':
                    value = (np.tan(zenith)**2.)*np.sin(2*par_angle)
                elif map_type == 'parallactic':
                    value = par_angle
                else:
                    value = self.table[map_type][ind]

                for j, op in enumerate(map_operation_list[i]):
                    if op == OP_SUM:
                        map_values_list[i][use, j] += value
                    elif op == OP_MEAN:
                        map_values_list[i][use, j] += value
                    elif op == OP_WMEAN:
                        map_values_list[i][use, j] += pixel_weights*value
                    elif op == OP_MIN:
                        map_values_list[i][use, j] = np.fmin(map_values_list[i][use, j],
                                                                    value)
                    elif op == OP_MAX:
                        map_values_list[i][pixels_off, j] = np.fmax(map_values_list[i][use, j],
                                                                    value)

        # Finish computations and save
        valid_pixels_use, = np.where(nexp > 0)

        if len(valid_pixels_use) == 0:
            return

        for i, map_type in enumerate(self.config.map_types.keys()):
            for j, op in enumerate(map_operation_list[i]):
                if op == OP_NONE:
                    # We do not need to write this map (it is already there)
                    continue
                elif op == OP_MEAN:
                    map_values_list[i][valid_pixels_use, j] /= nexp[valid_pixels_use]
                elif op == OP_WMEAN:
                    if map_type == 'maglimit':
                        maglimits = self._compute_maglimits(weights[valid_pixels_use])
                        map_values_list[i][valid_pixels_use, j] = maglimits
                    else:
                        map_values_list[i][valid_pixels_use, j] /= weights[valid_pixels_use]
                fname = map_fname_list[i][j]

                if map_type == 'coverage':
                    m = healsparse.HealSparseMap.make_empty(self.config.nside_coverage,
                                                            nside_sparse=self.config.nside,
                                                            dtype=map_values_list[i][:, j].dtype,
                                                            wide_mask_maxbits=1)
                    m.set_bits_pix(valid_pixels[valid_pixels_use], [0])
                else:
                    m = healsparse.HealSparseMap.make_empty(self.config.nside_coverage,
                                                            nside_sparse=self.config.nside,
                                                            dtype=map_values_list[i][:, j].dtype)
                    m[valid_pixels[valid_pixels_use]] = map_values_list[i][valid_pixels_use, j]
                m.write(fname)

    def build_region_input_map(self, indices, tilename=None, hpix=None):
        """
        Build input map for a given tile or hpix.
        Must specify tilename or hpix.

        Parameters
        ----------
        indices : `np.ndarray`
           Indices of WCS table
        tilename : `str`, optional
           Name of tile
        hpix : `int`, optional
           Healpix number

        Returns
        -------
        input_map : `healsparse.HealSparseMap`
           Wide mask map
        """
        if tilename is None and hpix is None:
            raise RuntimeError("Must specify one of tilename or hpix.")
        if tilename is not None and hpix is not None:
            raise RuntimeError("Must specify one of tilename or hpix.")

        region_input_map = healsparse.HealSparseMap.make_empty(nside_coverage=self.nside_coverage_region,
                                                               nside_sparse=self.config.nside,
                                                               dtype=healsparse.WIDE_MASK,
                                                               wide_mask_maxbits=len(indices)*2)
        metadata = {}
        for i, ind in enumerate(indices):
            # Everything needs to be done with 2 amps
            bit_a = i*2
            bit_b = i*2 + 1

            metadata['B%04dCCD' % (bit_a)] = self.table['ccdnum'][ind]
            metadata['B%04dCCD' % (bit_b)] = self.table['ccdnum'][ind]
            metadata['B%04dEXP' % (bit_a)] = self.table['expnum'][ind]
            metadata['B%04dEXP' % (bit_b)] = self.table['expnum'][ind]
            metadata['B%04dAMP' % (bit_a)] = 'A'
            metadata['B%04dAMP' % (bit_b)] = 'B'
            metadata['B%04dWT' % (bit_a)] = self._compute_weight('a', ind)
            metadata['B%04dWT' % (bit_b)] = self._compute_weight('b', ind)

            wcs = self.wcs_list[ind]

            if wcs is None:
                continue

            x_coords_a = np.array([self.config.amp_boundary,
                                   self.config.amp_boundary,
                                   self.table['naxis1'][ind] - self.config.border,
                                   self.table['naxis1'][ind] - self.config.border])
            x_coords_b = np.array([self.config.border,
                                   self.config.border,
                                   self.config.amp_boundary,
                                   self.config.amp_boundary])

            y_coords = np.array([self.config.border,
                                 self.table['naxis2'][ind] - self.config.border,
                                 self.table['naxis2'][ind] - self.config.border,
                                 self.config.border])

            ra_a, dec_a = wcs.image2sky(x_coords_a, y_coords)
            ra_b, dec_b = wcs.image2sky(x_coords_b, y_coords)

            poly_a = healsparse.Polygon(ra=ra_a, dec=dec_a, value=[bit_a])
            poly_b = healsparse.Polygon(ra=ra_b, dec=dec_b, value=[bit_b])

            # Check if we have additional masking
            if (self.streak_table is not None or self.bleed_table is not None or \
                    self.satstar_table is not None):
                poly_map_a = poly_a.get_map_like(region_input_map)
                poly_map_b = poly_b.get_map_like(region_input_map)

                mask_reg_list = []

                if self.streak_table is not None:
                    # This may need to be optimized
                    sinds, = np.where((self.streak_table['expnum'] == self.table['expnum'][ind]) &
                                      (self.streak_table['ccdnum'] == self.table['ccdnum'][ind]))
                    for sind in sinds:
                        mask_reg_list.append(self._get_maskpoly_from_row(self.streak_table[sind]))

                if self.bleed_table is not None:
                    binds, = np.where((self.bleed_table['expnum'] == self.table['expnum'][ind]) &
                                      (self.bleed_table['ccdnum'] == self.table['ccdnum'][ind]))
                    for bind in binds:
                        mask_reg_list.append(self._get_maskpoly_from_row(self.bleed_table[bind]))

                if self.satstar_table is not None:
                    sinds, = np.where((self.satstar_table['expnum'] == self.table['expnum'][ind]) &
                                      (self.satstar_table['ccdnum'] == self.table['ccdnum'][ind]))
                    for sind in sinds:
                        mask_reg_list.append(self._get_maskcircle_from_row(self.satstar_table[sind]))
                mask_map = healsparse.HealSparseMap.make_empty(nside_coverage=self.nside_coverage_region,
                                                               nside_sparse=self.config.nside,
                                                               dtype=np.uint8)
                healsparse.realize_geom(mask_reg_list, mask_map)
                poly_map_a.apply_mask(mask_map)
                poly_map_b.apply_mask(mask_map)

                pixels_a = poly_map_a.valid_pixels
                pixels_b = poly_map_b.valid_pixels
            else:
                # With no masking we can do a slightly faster version direct
                # with the pixels
                pixels_a = poly_a.get_pixels(nside=self.config.nside)
                pixels_b = poly_b.get_pixels(nside=self.config.nside)

            # Check for bad amps
            if int(self.table['ccdnum'][ind]) in list(self.config.bad_amps):
                ba = self.config.bad_amps[int(self.table['ccdnum'][ind])]
                for b in ba:
                    if b.lower() == 'a':
                        pixels_a = np.array([], dtype=np.int64)
                    elif b.lower() == 'b':
                        pixels_b = np.array([], dtype=np.int64)

            if tilename is not None:
                tind, = np.where(self.tile_info['tilename'] == tilename)
                for pixels, bit in zip([pixels_a, pixels_b], [bit_a, bit_b]):
                    pixra, pixdec = hp.pix2ang(self.config.nside, pixels, lonlat=True, nest=True)
                    if self.tile_info['crossra0'][tind] == 'Y':
                        # Special for cross-ra0, where uramin will be very large
                        uramin = self.tile_info['uramin'][tind] - 360.0
                        pixra_rot = pixra.copy()
                        hi, = np.where(pixra > 180.0)
                        pixra_rot[hi] -= 360.0
                        ok = ((pixra_rot > uramin) &
                              (pixra_rot < self.tile_info['uramax'][tind]) &
                              (pixdec > self.tile_info['udecmin'][tind]) &
                              (pixdec <= self.tile_info['udecmax'][tind]))
                    else:
                        ok = ((pixra > self.tile_info['uramin'][tind]) &
                              (pixra <= self.tile_info['uramax'][tind]) &
                              (pixdec > self.tile_info['udecmin'][tind]) &
                              (pixdec <= self.tile_info['udecmax'][tind]))
                    region_input_map.set_bits_pix(pixels[ok], [bit])
            else:
                # healpix mode
                bit_shift = 2*int(np.round(np.log2(self.config.nside/self.config.nside_run)))
                npixels = 2**bit_shift
                pixel_min = hpix*npixels
                pixel_max = (hpix + 1)*npixels - 1
                for pixels, bit in zip([pixels_a, pixels_b], [bit_a, bit_b]):
                    ok = ((pixels >= pixel_min) & (pixels <= pixel_max))
                    region_input_map.set_bits_pix(pixels[ok], [bit])

        region_input_map.metadata = metadata

        return region_input_map

    def _compute_weight(self, ampname, ind):
        """
        Compute the weight from a given amp

        Parameters
        ----------
        ampname : `str`
           Amp name, 'a' or 'b'
        ind : `int`
           Index in table to compute
        """
        return 1.0/(self.table['skyvar' + ampname][ind] *
                    100.**((self.config.zp_global -
                           self.table[self.config.magzp_field][ind])/2.5))

    def _compute_maglimits(self, weights):
        """
        Compute the maglimit from summed weights.

        Parameters
        ----------
        weights : `np.ndarray`
           Array of summed weights

        Returns
        -------
        maglimits : `np.ndarray`
           Array of mag limits
        """
        maglimits = (self.config.zp_global -
                     2.5*np.log10(10.*np.sqrt(np.pi) *
                                  self.config.maglim_aperture/(2.*self.config.arcsec_per_pix)) -
                     2.5*np.log10(1./np.sqrt(weights)))
        return maglimits

    def _compute_zenith_and_par_angles(self, loc, mjd, ra, dec):
        """
        Compute the zenith angle for a given ra/dec

        Parameters
        ----------
        loc : `astropy.coordinates.EarthLocation`
        mjd : `float`
        ra : `float`
           RA in degrees
        dec : `float`
           Dec in degrees

        Returns
        -------
        zenith_angle : `float`
           Zenith angle in radians.
        parallactic_angle : `float`, optional
           Parallactic angle in radians.
        """
        t = Time(mjd, format='mjd', location=loc)
        lst = t.sidereal_time('apparent')
        ha = lst - ra*units.degree

        c_ra = ra*coord.degrees
        c_dec = dec*coord.degrees
        c_ha = ha.to_value(units.degree)*coord.degrees
        c_lat = self.config.latitude*coord.degrees
        c_zenith = coord.CelestialCoord(c_ha + c_ra, c_lat)
        c_pointing = coord.CelestialCoord(c_ra, c_dec)
        zenith_angle = c_pointing.distanceTo(c_zenith).rad

        c_NCP = coord.CelestialCoord(0.0*coord.degrees, 90.0*coord.degrees)
        parallactic_angle = c_pointing.angleBetween(c_NCP, c_zenith).rad

        return zenith_angle, parallactic_angle

    def _compute_airmass(self, zenith):
        """
        Compute the airmass for a list of zenith angles.
        Computed using simple expansion formula.

        Parameters
        ----------
        zenith : `np.ndarray`
           Zenith angle(s), radians

        Returns
        -------
        airmass : `np.ndarray`
        """
        secz = 1./np.cos(zenith)
        airmass = (secz -
                   0.0018167*(secz - 1.0) -
                   0.002875*(secz - 1.0)**2.0 -
                   0.0008083*(secz - 1.0)**3.0)
        return airmass

    def _get_maskpoly_from_row(self, table_row):
        """
        Get a mask polygon from a table row

        Parameters
        ----------
        table_row : `np.ndarray`
           Row of a mask table with RAs and Decs

        Returns
        -------
        maskpoly : `healsparse.Polygon`
        """
        mask_poly = healsparse.Polygon(ra=np.array([table_row['ra_1'],
                                                    table_row['ra_2'],
                                                    table_row['ra_3'],
                                                    table_row['ra_4']]),
                                       dec=np.array([table_row['dec_1'],
                                                     table_row['dec_2'],
                                                     table_row['dec_3'],
                                                     table_row['dec_4']]),
                                       value=1)
        return mask_poly

    def _get_maskcircle_from_row(self, table_row):
        """
        Get a mask circle from a table row

        Parameters
        ----------
        table_row : `np.ndarray`
           Row of a mask table with RAs and Decs

        Returns
        -------
        maskcircle : `healsparse.Circle`
        """
        maskcircle = healsparse.Circle(ra=table_row['ra'],
                                       dec=table_row['dec'],
                                       radius=table_row['radius']/3600.,
                                       value=1)
        return maskcircle

