import unittest
import os
import tempfile
from collections import OrderedDict

import decasu

import decasu_test_base


ROOT = os.path.abspath(os.path.dirname(__file__))


skip_lsst_tests = True
try:
    import lsst.obs.lsst  # noqa: F401
    skip_lsst_tests = False
except ImportError:
    skip_lsst_tests = True


@unittest.skipIf(skip_lsst_tests, "Rubin Science Pipelines not installed.")
class HpixLSSTTestCase(decasu_test_base.DecasuTestBase):
    """
    Tests for running healpixels using the LSST code.
    """
    def test_hpix_lsst(self):
        """
        Test several hpixels
        """
        band = 'r'

        self.test_dir = tempfile.mkdtemp(dir=ROOT, prefix='TestHpixLSST-')

        config = decasu.Configuration.load_yaml(os.path.join(ROOT, 'configs',
                                                             'config_hpix_lsst.yaml'))

        dbfile = os.path.join(ROOT, 'data', 'baseline_v2.0_10yrs_test_data.db')

        mapper = decasu.MultiHealpixMapper(config, self.test_dir, ncores=1)
        mapper(dbfile, bands=[band],
               clear_intermediate_files=False)

        # Look at 549
        expected_dict = OrderedDict()
        expected_dict['inputs'] = [200, 45]
        expected_dict['airmass_max'] = [1.05, 1.45, 'float64']
        expected_dict['airmass_min'] = [1.05, 1.45, 'float64']
        expected_dict['airmass_wmean'] = [1.05, 1.45, 'float64']
        expected_dict['exptime_sum'] = [29.0, 61.0, 'float64']
        expected_dict['nexp_sum'] = [0, 3, 'int32']

        self.check_expected_maps_hpix(expected_dict, 8, 549, 'r')

        # Look at consolidated map
        expected_dict = OrderedDict()
        expected_dict['airmass_max'] = [1.05, 1.45, 'float64']
        expected_dict['airmass_min'] = [1.05, 1.45, 'float64']
        expected_dict['airmass_wmean'] = [1.05, 1.45, 'float64']
        expected_dict['exptime_sum'] = [29.0, 61.0, 'float64']
        expected_dict['nexp_sum'] = [0, 3, 'int32']

        self.check_expected_maps_consolidated(expected_dict, 'r')


if __name__ == '__main__':
    unittest.main()
