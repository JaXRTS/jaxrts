from sphinx_gallery.scrapers import matplotlib_scraper


class matplotlib_svg_scraper:
    def __repr__(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        return matplotlib_scraper(*args, format="svg", **kwargs)

