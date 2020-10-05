import unittest
import os
import tempfile
from collections import OrderedDict

import decasu

import decasu_test_base


class TilenameTestCase(decasu_test_base.DecasuTestBase):
    """
    Tests for running a single tilename.
    """
    def test_tilename(self):
        """
        Test a single tilename
        """
        tilenames = ['DES0003-5457', 'DES2358-5457']
        band = 'i'

        self.test_dir = tempfile.mkdtemp(dir='./', prefix='TestTilename-')

        config = decasu.Configuration.load_yaml(os.path.join('./', 'configs',
                                                             'config_tilename.yaml'))

        coaddtilefile = os.path.join('./', 'data', 'y3a2_testing_coadd_tiles_and_geom.fits.gz')
        imagefile = os.path.join('./', 'data', 'y3a2_testing_coadd_input_image_table.fits.gz')
        bleedtrailfile = os.path.join('./', 'data', 'y3a2_testing_bleedtrails.fits.gz')
        streakfile = os.path.join('./', 'data', 'y3a2_testing_streaks.fits.gz')
        starfile = os.path.join('./', 'data', 'y3a2_testing_satstars.fits.gz')

        mapper = decasu.MultiTileMapper(config, self.test_dir, ncores=1)
        mapper(coaddtilefile, [imagefile], band, coaddtiles=tilenames,
               bleedtrailfiles=[bleedtrailfile], streakfiles=[streakfile],
               starfiles=[starfile], clear_intermediate_files=False, clobber=False)

        # Do each of the tiles in turn, and compare the consolidated one to mangle

        # DES0003-5457
        expected_dict = OrderedDict()
        expected_dict['inputs'] = [140, 18]
        expected_dict['airmass_max'] = [1.08, 1.45, 'float64']
        expected_dict['airmass_min'] = [1.08, 1.45, 'float64']
        expected_dict['airmass_wmean'] = [1.08, 1.45, 'float64']
        expected_dict['dcr_dra_wmean'] = [-0.15, 1.02, 'float64']
        expected_dict['dcr_ddec_wmean'] = [0.15, 0.50, 'float64']
        expected_dict['dcr_e1_wmean'] = [-1.02, 0.22, 'float64']
        expected_dict['dcr_e2_wmean'] = [-0.13, 0.45, 'float64']
        expected_dict['fwhm_wmean'] = [2.50, 4.40, 'float64']
        expected_dict['maglimit_wmean'] = [22.37, 23.60, 'float64']
        expected_dict['nexp_sum'] = [0, 7, 'int32']
        expected_dict['skybrite_wmean'] = [1830.0, 5550.0, 'float64']
        expected_dict['skysigma_wmean'] = [51.4, 77.2, 'float64']

        self.check_expected_maps_tile(expected_dict, 'DES0003-5457', band)

        # DES2358-5457
        expected_dict = OrderedDict()
        expected_dict['inputs'] = [144, 18]
        expected_dict['airmass_max'] = [1.08, 1.39, 'float64']
        expected_dict['airmass_min'] = [1.08, 1.39, 'float64']
        expected_dict['airmass_wmean'] = [1.08, 1.39, 'float64']
        expected_dict['dcr_dra_wmean'] = [-0.15, 0.94, 'float64']
        expected_dict['dcr_ddec_wmean'] = [0.20, 0.47, 'float64']
        expected_dict['dcr_e1_wmean'] = [-0.82, 0.22, 'float64']
        expected_dict['dcr_e2_wmean'] = [-0.12, 0.46, 'float64']
        expected_dict['fwhm_wmean'] = [2.50, 4.02, 'float64']
        expected_dict['maglimit_wmean'] = [22.37, 23.60, 'float64']
        expected_dict['nexp_sum'] = [0, 7, 'int32']
        expected_dict['exptime_sum'] = [80.0, 640.0, 'float64']
        expected_dict['skybrite_wmean'] = [2510.0, 5565.0, 'float64']
        expected_dict['skysigma_wmean'] = [50.1, 78.7, 'float64']

        self.check_expected_maps_tile(expected_dict, 'DES2358-5457', band)

        self.check_mangle_map('y3a2_%s_mangle_4096' % (band),
                              band, 'airmass', 'wmean', 0.02, 0.01)
        self.check_mangle_map('y3a2_%s_mangle_4096' % (band),
                              band, 'exptime', 'sum', 0.10, 0.01)
        self.check_mangle_map('y3a2_%s_mangle_4096' % (band),
                              band, 'fwhm', 'wmean', 0.025, 0.01, scale=0.263)
        self.check_mangle_map('y3a2_%s_mangle_4096' % (band),
                              band, 'maglimit', 'wmean', 0.03, 0.1)


if __name__ == '__main__':
    unittest.main()
