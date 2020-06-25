from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import yaml
import os
import numpy as np


@dataclass
class Configuration(object):
    """
    Decasu configuration object.
    """
    # Mandatory fields

    # Optional fields
    nside: int = 32768
    nside_coverage: int = 32
    ctype1: str = 'RA---TPV'
    ctype2: str = 'DEC--TPV'
    border: int = 15

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
