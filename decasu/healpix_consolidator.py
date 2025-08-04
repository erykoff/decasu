import os
import healsparse


class HealpixConsolidator:
    """
    Consolidate several maps into one.

    Parameters
    ----------
    config : `Configuration`
       decasu configuration object
    clear_intermediate_files : `bool`
       Clear input files when done?
    """
    def __init__(self, config, clear_intermediate_files, make_map_images=False):
        self.config = config
        self.clear_intermediate_files = clear_intermediate_files
        self.make_map_images = make_map_images

    def __call__(self, fname, mapfiles, descr):
        """
        Consolidate a list of mapfiles, and delete input mapfiles
        if clear_intermediate_files is True.

        Parameters
        ----------
        fname : `str`
            Output filename
        mapfiles : `list`
            Input list of files
        descr : `str`
            Description string.
        """
        print("Consolidating %d maps into %s" % (len(mapfiles), fname))
        healsparse.cat_healsparse_files(mapfiles, fname)

        if self.make_map_images:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            import skyproj

            m = healsparse.HealSparseMap.read(fname)

            fig = Figure(figsize=(10, 6))
            FigureCanvasAgg(fig)
            ax = fig.add_subplot(111)

            sp = skyproj.McBrydeSkyproj(ax=ax)
            sp.draw_hspmap(m, zoom=True)
            sp.draw_colorbar(label=descr)
            skyprojfile = fname.replace(".hsp", "_skyproj.png")
            fig.savefig(skyprojfile)

        if self.clear_intermediate_files:
            for f in mapfiles:
                os.unlink(f)
