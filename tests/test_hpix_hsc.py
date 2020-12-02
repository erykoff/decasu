import unittest
import os
import tempfile
from collections import OrderedDict

import decasu

import decasu_test_base


class HpixHscTestCase(decasu_test_base.DecasuTestBase):
    """
    Tests for running a healpixels using the HSC code.
    """
    def test_hpix_hsc(self):
        """
        Test a several hpixels
        """
        band = 'r'

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestHpixHSC-')

        config = decasu.Configuration.load_yaml(os.path.join('./', 'configs',
                                                             'config_hpix_hsc.yaml'))

        imagefile = os.path.join('./', 'data', 'S16A_WIDE_frames_test.fits.gz')

        mapper = decasu.MultiHealpixMapper(config, self.test_dir, ncores=1)
        mapper(imagefile, bands=[band],
               clear_intermediate_files=False)

        # Look at 277
        expected_dict = OrderedDict()
        expected_dict['inputs'] = [200, 31]
        expected_dict['airmass_max'] = [1.14, 1.39, 'float64']
        expected_dict['airmass_min'] = [1.14, 1.39, 'float64']
        expected_dict['airmass_wmean'] = [1.14, 1.39, 'float64']
        expected_dict['dcr_dra_wmean'] = [0.38, 0.85, 'float64']
        expected_dict['dcr_ddec_wmean'] = [0.38, 0.48, 'float64']
        expected_dict['dcr_e1_wmean'] = [-0.50, 0.01, 'float64']
        expected_dict['dcr_e2_wmean'] = [0.30, 0.78, 'float64']
        expected_dict['seeing_wmean'] = [-10000.0, 10.6, 'float64']
        expected_dict['maglim_wmean'] = [17.9, 19.48, 'float64']
        expected_dict['nexp_sum'] = [0, 6, 'int32']
        expected_dict['exptime_sum'] = [29.0, 391.0, 'float64']
        expected_dict['skylevel_wmean'] = [273.0, 1604.0, 'float64']
        expected_dict['skylevel_wmean-scaled'] = [290.0, 1590.0, 'float64']
        expected_dict['sigma_sky_wmean'] = [8.4, 117.0, 'float64']

        self.check_expected_maps_hpix(expected_dict, 8, 277, band)

        # And consolidated
        expected_dict = OrderedDict()
        expected_dict['airmass_max'] = [1.14, 1.40, 'float64']
        expected_dict['airmass_min'] = [1.14, 1.39, 'float64']
        expected_dict['airmass_wmean'] = [1.14, 1.39, 'float64']
        expected_dict['dcr_dra_wmean'] = [0.39, 0.85, 'float64']
        expected_dict['dcr_ddec_wmean'] = [0.39, 0.47, 'float64']
        expected_dict['dcr_e1_wmean'] = [-0.51, 0.01, 'float64']
        expected_dict['dcr_e2_wmean'] = [0.31, 0.80, 'float64']
        expected_dict['seeing_wmean'] = [-10000.0, 10.6, 'float64']
        expected_dict['maglim_wmean'] = [17.8, 19.6, 'float64']
        expected_dict['nexp_sum'] = [0, 6, 'int32']
        expected_dict['exptime_sum'] = [29.0, 391.0, 'float64']
        expected_dict['skylevel_wmean'] = [264.0, 1604.0, 'float64']
        expected_dict['skylevel_wmean-scaled'] = [255.0, 1590.0, 'float64']
        expected_dict['sigma_sky_wmean'] = [8.4, 116.5, 'float64']

        self.check_expected_maps_consolidated(expected_dict, band)


if __name__ == '__main__':
    unittest.main()
