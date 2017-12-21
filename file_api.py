from openslide import AbstractSlide
import openslide
import KFBReading.kfbslide as kfbslide

class AllSlide(AbstractSlide):
    @property
    def level_dimensions(self):
        return self._osr.level_dimensions

    def read_region(self, location, level, size):
        return self._osr.read_region(location, level, size).convert('RGB')

    @property
    def level_count(self):
        return self._osr.level_count

    def __init__(self, filename):
        super(AllSlide, self).__init__()
        self.filename = filename
	print filename
        if filename.endswith('.svs'):
            self._osr = openslide.open_slide(filename)
        elif filename.endswith('.kfb'):
            self._osr = kfbslide.KfbSlide(filename)
        else:
            print('not support ', filename.split('.')[-1],'!')
            exit(0)

def get_mask_info(filename):
    if filename.find('_&_')!=-1:
        return filename.split('_&_')
    else:
        return filename.split('_')
