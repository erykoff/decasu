import unittest
import os
import numpy as np
import hpgeom as hpg
import numpy.testing as testing
import shutil

import healsparse


ROOT = os.path.abspath(os.path.dirname(__file__))


class DecasuTestBase(unittest.TestCase):
    def setUp(self):
        self.test_dir = None

    def tearDown(self):
        if self.test_dir is not None:
            if os.path.exists(self.test_dir):
                shutil.rmtree(self.test_dir, True)

    def check_expected_maps_tile(self, expected_dict, tilename, band, time_bin=-1):
        """
        Check for expected maps, ranges, types for a tilename.

        Parameters
        ----------
        expected_dict : `OrderedDict`
            Ordered dictionary with keys of map names, values of [min, max].
            The first item must be the tile input with [nccd, width_expected].
        tilename : `str`
            Name of tile
        band : `str`
            Band name
        time_bin : `int`, optional
            Bin of time to be processing.
        """
        if time_bin < 0:
            outbase = 'testing'
        else:
            outbase = 'testing-%d' % (time_bin)

        mod_times = []
        for em in expected_dict:
            map_name = os.path.join(self.test_dir, tilename, '%s_%s_%s_%s.hsp'
                                    % (outbase, tilename, band, em))
            self.assertTrue(os.path.isfile(map_name))
            mod_times.append(os.path.getmtime(map_name))
            m = healsparse.HealSparseMap.read(map_name)

            valid_pixels = np.sort(m.valid_pixels)
            if em == 'inputs':
                input_valid_pixels = valid_pixels
                metadata = m.metadata

                # Check the metadata
                self.assertTrue(metadata['WIDEMASK'])
                self.assertEqual(metadata['WWIDTH'], expected_dict[em][1])
                nccd = 0
                nexp = 0
                namp = 0
                nwt = 0
                for i in range(200):
                    if 'B%04dCCD' % (i) in metadata:
                        nccd += 1
                    if 'B%04dEXP' % (i) in metadata:
                        nexp += 1
                    if 'B%04dAMP' % (i) in metadata:
                        namp += 1
                    if 'B%04dWT' % (i) in metadata:
                        nwt += 1
                self.assertEqual(nccd, expected_dict[em][0])
                self.assertEqual(nexp, nccd)
                self.assertEqual(namp, nccd)
                self.assertEqual(nwt, nccd)
            else:
                # Make sure we have the same valid pixels as the input map
                testing.assert_array_equal(valid_pixels, input_valid_pixels)

                self.assertEqual(m.dtype.name, expected_dict[em][2])
                self.assertGreater(np.min(m[valid_pixels]),
                                   expected_dict[em][0])
                self.assertLess(np.max(m[valid_pixels]),
                                expected_dict[em][1])

        return mod_times

    def check_mangle_map(self, manglebase, band, mapstr, opstr, precision,
                         max_outlier_frac, scale=1.0, delta=False):
        """
        Check a rendered map vs a mangle map, with tolerances.

        Parameters
        ----------
        manglebase : `str`
           Base of the mangle map name.
        band : `str`
           Name of the band to compaire
        mapstr : `str`
           String representation of the mapped quantity
        opstr : `str`
           String representation of the map operation
        precision : `float`
           Max percentage deviation for matched pixels
        max_outlier_frac : `float`
           Max fraction of matched pixels that are outliers
        scale : `float`, optional
           Scale the decasu values by this factor before comparing
        delta : `bool`, optional
           Compute delta instead of ratio
        """
        # Read in the mangle map
        mangle_map = healsparse.HealSparseMap.read(os.path.join(ROOT, 'data',
                                                                '%s_%s.hsp' %
                                                                (manglebase, mapstr)))
        nside_mangle = mangle_map.nside_sparse

        decasu_map = healsparse.HealSparseMap.read(os.path.join(self.test_dir,
                                                                'testing_%s_%s_%s.hsp' %
                                                                (band, mapstr, opstr)))
        decasu_map_dg = decasu_map.degrade(nside_mangle)

        vpix_decasu = decasu_map_dg.valid_pixels
        vpix_mangle = mangle_map.valid_pixels

        values_mangle = mangle_map[vpix_decasu]
        values_decasu = decasu_map_dg[vpix_decasu]*scale

        gd, = np.where(values_mangle > hpg.UNSEEN)
        gd2, = np.where(decasu_map_dg[vpix_mangle] > hpg.UNSEEN)

        # Check that this is above a certain threshold...
        # This first test checks that most of the pixels in the decasu
        # map are in the mangle map
        self.assertGreater(gd.size/vpix_decasu.size, 0.95)
        # This second test checks that most of the pixels in the mangle
        # map are in the decasu map
        self.assertGreater(gd2.size/vpix_mangle.size, 0.95)

        if delta:
            deltas = values_decasu[gd] - values_mangle[gd]
            outliers, = np.where(np.abs(deltas) > precision)
        else:
            ratios = values_decasu[gd]/values_mangle[gd]
            outliers, = np.where((ratios < (1.0 - precision)) |
                                 (ratios > (1.0 + precision)))

        self.assertLess(outliers.size/gd.size, max_outlier_frac)

    def check_expected_maps_hpix(self, expected_dict, nside, hpix, band, check_amp=False, time_bin=-1):
        """
        Check for expected maps, ranges, types for a tilename.

        Parameters
        ----------
        expected_dict : `OrderedDict`
            Ordered dictionary with keys of map names, values of [min, max].
            The first item must be the tile input with [nccd, width_expected].
        nside : `int`
            Healpix nside
        hpix : `int`
            Healpixel number
        band : `str`
            Band name
        check_amp : `bool`, optional
            Check that the amplifier is set
        time_bin : `int`, optional
            Bin of time to be processing.
        """
        if time_bin < 0:
            outbase = 'testing'
        else:
            outbase = 'testing-%d' % (time_bin)

        mod_times = []
        for em in expected_dict:
            subdir = '%d_%05d' % (nside, hpix)
            map_name = os.path.join(self.test_dir, subdir, '%s_%d_%05d_%s_%s.hsp'
                                    % (outbase, nside, hpix, band, em))
            self.assertTrue(os.path.isfile(map_name))
            mod_times.append(os.path.getmtime(map_name))
            m = healsparse.HealSparseMap.read(map_name)

            valid_pixels = np.sort(m.valid_pixels)
            if em == 'inputs':
                input_valid_pixels = valid_pixels
                metadata = m.metadata

                # Check the metadata
                self.assertTrue(metadata['WIDEMASK'])
                self.assertEqual(metadata['WWIDTH'], expected_dict[em][1])
                nccd = 0
                nexp = 0
                namp = 0
                nwt = 0
                for i in range(200):
                    if 'B%04dCCD' % (i) in metadata:
                        nccd += 1
                    if 'B%04dEXP' % (i) in metadata:
                        nexp += 1
                    if 'B%04dAMP' % (i) in metadata:
                        namp += 1
                    if 'B%04dWT' % (i) in metadata:
                        nwt += 1
                self.assertEqual(nccd, expected_dict[em][0])
                self.assertEqual(nexp, nccd)
                if check_amp:
                    self.assertEqual(namp, nccd)
                self.assertEqual(nwt, nccd)
            else:
                # Make sure we have the same valid pixels as the input map
                testing.assert_array_equal(valid_pixels, input_valid_pixels)

                self.assertEqual(m.dtype.name, expected_dict[em][2])
                self.assertGreater(np.min(m[valid_pixels]),
                                   expected_dict[em][0])
                self.assertLess(np.max(m[valid_pixels]),
                                expected_dict[em][1])

        return mod_times

    def check_expected_maps_consolidated(self, expected_dict, band, time_bin=-1):
        """
        Check for expected consolidated maps, ranges, types.

        Parameters
        ----------
        expected_dict : `OrderedDict`
            Ordered dictionary with keys of map names, values of [min, max].
        band : `str`
            Band name
        time_bin : `int`, optional
            Bin of time to be processing.
        """
        if time_bin < 0:
            outbase = 'testing'
        else:
            outbase = 'testing-%d' % (time_bin)

        mod_times = []
        first_valid_pixels = None
        for em in expected_dict:
            map_name = os.path.join(self.test_dir, '%s_%s_%s.hsp'
                                    % (outbase, band, em))
            self.assertTrue(os.path.isfile(map_name))
            mod_times.append(os.path.getmtime(map_name))
            m = healsparse.HealSparseMap.read(map_name)

            valid_pixels = np.sort(m.valid_pixels)

            if first_valid_pixels is None:
                first_valid_pixels = valid_pixels

            # Make sure we have the same valid pixels as the first map
            testing.assert_array_equal(valid_pixels, first_valid_pixels)

            self.assertEqual(m.dtype.name, expected_dict[em][2])
            self.assertEqual(m.dtype.name, expected_dict[em][2])

            self.assertGreater(np.min(m[valid_pixels]),
                               expected_dict[em][0])
            self.assertLess(np.max(m[valid_pixels]),
                            expected_dict[em][1])

        return mod_times
