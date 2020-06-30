from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict
import yaml

from .utils import op_code_to_str


def _default_extra_fields():
    return {'ctype1': 'RA---TPV',
            'ctype2': 'DEC--TPV',
            'cunit1': 'deg',
            'cunit2': 'deg'}


@dataclass
class Configuration(object):
    """
    Decasu configuration object.
    """
    # Mandatory fields
    outbase: str
    map_types: dict

    # Optional fields
    nside: int = 32768
    nside_coverage: int = 32
    nside_run: int = 8
    ncore: int = 1
    extra_fields: Dict[str, str] = field(default_factory=_default_extra_fields)
    border: int = 15
    amp_boundary: int = 1024
    arcsec_per_pix: float = 0.263
    maglim_aperture: float = 2.0
    zp_global: float = 30.0
    zp_sign_swap: bool = False
    magzp_field: str = 'mag_zero'

    def __post_init__(self):
        self._validate()

    def _validate(self):
        pass

    @classmethod
    def load_yaml(cls, configfile):
        """
        Load a yaml file into the config

        Parameters
        ----------
        configfile : `str`
           Filename of yaml file

        Returns
        -------
        config : `Configuration`
        """
        with open(configfile) as f:
            _config = yaml.load(f, Loader=yaml.SafeLoader)

        return cls(**_config)

    def healpix_relpath(self, hpix):
        """
        Compute relative path for a given healpix

        Parameters
        ----------
        hpix : `int`

        Returns
        -------
        relpath : `str`
        """
        return '%d_%05d' % (self.nside_run, hpix)

    def healpix_map_filename(self, band, hpix, map_type, operation):
        """
        Compute healpix map filename.

        Parameters
        ----------
        band : `str`
           Name of band
        hpix : `int`
           Number of healpix
        map_type : `str`
           Type of map
        operation : `int`
           Enumerated operation type

        Returns
        -------
        filename : `str`
        """
        return "%s_%d_%05d_%s_%s_%s.hs" % (self.outbase,
                                           self.nside_run,
                                           hpix,
                                           band,
                                           map_type,
                                           op_code_to_str(operation))

    def healpix_map_filename_template(self, band, map_type, operation):
        """
        Compute healpix map template filename

        Parameters
        ----------
        band : `str`
           Name of band
        hpix : `int`
           Number of healpix
        map_type : `str`
           Type of map
        operation : `int`
           Enumerated operation type

        Returns
        -------
        template : `str`
        """
        return "%s_%d_?????_%s_%s_%s.hs" % (self.outbase,
                                            self.nside_run,
                                            band,
                                            map_type,
                                            op_code_to_str(operation))

    def map_filename(self, band, map_type, operation):
        """
        Compute full map filename.

        Parameters
        ----------
        band : `str`
           Name of band
        hpix : `int`
           Number of healpix
        map_type : `str`
           Type of map
        operation : `int`
           Enumerated operation type

        Returns
        -------
        filename : `str`
        """
        return "%s_%s_%s_%s.hs" % (self.outbase,
                                   band,
                                   map_type,
                                   op_code_to_str(operation))
