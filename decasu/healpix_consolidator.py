import os
import healsparse


class HealpixConsolidator(object):
    """
    Consolidate several maps into one.

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    clear_intermediate_files : `bool`
       Clear input files when done?
    """
    def __init__(self, config, clear_intermediate_files):
        self.config = config
        self.clear_intermediate_files = clear_intermediate_files

    def __call__(self, fname, mapfiles):
        """
        Consolidate a list of mapfiles, and delete input mapfiles
        if clear_intermediate_files is True.

        Parameters
        ----------
        fname : `str`
           Output filename
        mapfiles : `list`
           Input list of files
        """
        print("Consolidating %d maps into %s" % (len(mapfiles), fname))
        healsparse.cat_healsparse_files(mapfiles, fname)

        if self.clear_intermediate_files:
            for f in mapfiles:
                os.unlink(f)
