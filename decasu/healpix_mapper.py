import os
import numpy as np
import healsparse

from .utils import op_str_to_code
from .utils import OP_NONE, OP_SUM, OP_MEAN, OP_WMEAN, OP_MIN, OP_MAX
from . import decasu_globals


class HealpixMapper(object):
    """
    Map a single healpix pixel

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    """
    def __init__(self, config, outputpath):
        self.config = config
        self.outputpath = outputpath

    def __call__(self, hpix, indices):
        """
        Run all the configured maps for a given hpix (nside self.config.nside_run).

        Parameters
        ----------
        hpix : `int`
           Healpix to run
        indices : `np.ndarray`
           Indices of table/wcs_list in the given healpix
        """
        self.table = decasu_globals.table
        self.wcs_list = decasu_globals.wcs_list

        print("Computing maps for pixel %d with %d inputs" % (hpix, len(indices)))

        # Create the path for the files if necessary
        os.makedirs(os.path.join(self.outputpath,
                                 self.config.healpix_relpath(hpix)),
                    exist_ok=True)

        bit_shift = 2*int(np.round(np.log2(self.config.nside / self.config.nside_run)))
        npixels = 2**bit_shift
        pixel_min = hpix*npixels
        pixel_max = (hpix + 1)*npixels - 1

        # Figure out how many maps we need to make and initialize memory
        map_values_list = []
        map_operation_list = []
        map_fname_list = []
        for map_type in self.config.map_types.keys():
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
                fname = os.path.join(self.outputpath,
                                     self.config.healpix_relpath(hpix),
                                     self.config.healpix_map_filename(self.table['band'][indices[0]],
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
                elif op == OP_WMEAN:
                    any_weights = True

        if not any_to_compute:
            # Everything is already there
            print("All maps for pixel %d are already computed.  Skipping..." % (hpix))
            return

        weights = np.zeros(npixels)
        nexp = np.zeros(npixels, dtype=np.int32)

        # Render and compute
        for ind in indices:
            wcs = self.wcs_list[ind]

            if wcs is None:
                continue

            # Need two amps
            x_coords_a = np.array([self.config.border,
                                   self.config.border,
                                   self.config.amp_boundary,
                                   self.config.amp_boundary])
            x_coords_b = np.array([self.config.amp_boundary,
                                   self.config.amp_boundary,
                                   self.table['naxis1'][ind] - self.config.border,
                                   self.table['naxis1'][ind] - self.config.border])

            y_coords = np.array([self.config.border,
                                 self.table['naxis2'][ind] - self.config.border,
                                 self.table['naxis2'][ind] - self.config.border,
                                 self.config.border])

            # Find all the pixels, per amp, that are within this coarse "run" pixel
            pixels_a = self._compute_box_pixels(wcs, x_coords_a, y_coords, pixel_min, pixel_max)
            pixels_b = self._compute_box_pixels(wcs, x_coords_b, y_coords, pixel_min, pixel_max)
            if len(pixels_a) == 0 and len(pixels_b) == 0:
                # Nothing in the big pixel (off edge)
                continue

            if any_weights:
                pixel_weights_a = self._compute_weight('a', ind)
                pixel_weights_b = self._compute_weight('b', ind)
            else:
                pixel_weights_a = 1.0
                pixel_weights_b = 1.0

            weights[pixels_a - pixel_min] += pixel_weights_a
            weights[pixels_b - pixel_min] += pixel_weights_b

            pixels_off = np.concatenate((pixels_a, pixels_b)) - pixel_min
            pixel_weights = np.concatenate((np.full(pixels_a.size, pixel_weights_a),
                                            np.full(pixels_b.size, pixel_weights_b)))
            nexp[pixels_off] += 1

            for i, map_type in enumerate(self.config.map_types.keys()):
                if map_type == 'nexp':
                    value = 1
                elif map_type == 'maglimit':
                    # We compute this below from the weights
                    value = 0.0
                elif map_type == 'coverage':
                    continue
                else:
                    value = self.table[map_type][ind]

                for j, op in enumerate(map_operation_list[i]):
                    if op == OP_SUM:
                        map_values_list[i][pixels_off, j] += value
                    elif op == OP_MEAN:
                        map_values_list[i][pixels_off, j] += value
                    elif op == OP_WMEAN:
                        map_values_list[i][pixels_off, j] += pixel_weights*value
                    elif op == OP_MIN:
                        map_values_list[i][pixels_off, j] = np.fmin(map_values_list[i][pixels_off, j],
                                                                    value)
                    elif op == OP_MAX:
                        map_values_list[i][pixels_off, j] = np.fmax(map_values_list[i][pixels_off, j],
                                                                    value)

        # Finish computations and save
        valid_pixels_off, = np.where(nexp > 0)

        if len(valid_pixels_off) == 0:
            return

        for i, map_type in enumerate(self.config.map_types.keys()):
            for j, op in enumerate(map_operation_list[i]):
                if op == OP_NONE:
                    # We do not need to write this map (it is already there)
                    continue
                elif op == OP_MEAN:
                    map_values_list[i][valid_pixels_off, j] /= nexp[valid_pixels_off]
                elif op == OP_WMEAN:
                    if map_type == 'maglimit':
                        maglimits = self._compute_maglimits(weights[valid_pixels_off])
                        map_values_list[i][valid_pixels_off, j] = maglimits
                    else:
                        map_values_list[i][valid_pixels_off, j] /= weights[valid_pixels_off]
                fname = map_fname_list[i][j]

                if map_type == 'coverage':
                    m = healsparse.HealSparseMap.make_empty(self.config.nside_coverage,
                                                            nside_sparse=self.config.nside,
                                                            dtype=map_values_list[i][:, j].dtype,
                                                            wide_mask_maxbits=1)
                    m.set_bits_pix(valid_pixels_off + pixel_min, [0])
                else:
                    m = healsparse.HealSparseMap.make_empty(self.config.nside_coverage,
                                                            nside_sparse=self.config.nside,
                                                            dtype=map_values_list[i][:, j].dtype)
                    m[valid_pixels_off + pixel_min] = map_values_list[i][valid_pixels_off, j]
                m.write(fname)

    def _compute_box_pixels(self, wcs, x_coords, y_coords, pixel_min, pixel_max):
        """
        Compute coverage pixels from the corner coordinates of a ccd/amp box

        Parameters
        ----------
        wcs : `esutil.wcsutil.WCS`
        x_coords : `np.ndarray`
           Array of x coordinates
        y_coords : `np.ndarray`
           Array of y coordinates
        pixel_min : `int`
           Minimum allowed pixel
        pixel_max : `int`
           Maximum allowed pixel

        Returns
        -------
        pixels : `np.ndarray`
        """
        ra, dec = wcs.image2sky(x_coords, y_coords)
        poly = healsparse.Polygon(ra=ra, dec=dec, value=1)
        pixels = poly.get_pixels(nside=self.config.nside)
        ok = ((pixels >= pixel_min) & (pixels <= pixel_max))
        return pixels[ok]

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
                    100.**(self.config.zp_global -
                           self.table[self.config.magzp_field][ind])/2.5)

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
                     2.5*np.log10(1./np.sqrt(1./weights)))
        return maglimits
