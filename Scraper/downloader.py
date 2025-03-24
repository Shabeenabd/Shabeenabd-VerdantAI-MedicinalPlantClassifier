import os, sys
import shutil
from pathlib import Path

try:
    from bing import Bing
except ImportError:  # Python 3
    from .bing import Bing


def download(query, limit=5, output_dir='dataset', adult_filter_off=True, 
force_replace=False, timeout=1, filter="", verbose=True):

    # engine = 'bing'
    if adult_filter_off:
        adult = 'off'
    else:
        adult = 'on'

    
    #image_dir = Path(output_dir)#.joinpath(query).absolute()
    image_dir=Path(output_dir).absolute()
  
    if force_replace:
        if Path.is_dir(image_dir):
            shutil.rmtree(image_dir)

    # check directory and create if necessary
    try:
        if not Path.is_dir(image_dir):
            print("n")
            Path.mkdir(image_dir, parents=True)

    except Exception as e:
        print('[Error]Failed to create directory.', e)
        sys.exit(1)
        
    print("[%] Downloading Images to {}".format(str(image_dir.absolute())))
    bing = Bing(query, limit, image_dir, adult, timeout, filter, verbose=False)
    bing.run()

classes=["dog","cat"]
if __name__ == '__main__':
    download(classes, output_dir="dataset", limit=4, timeout=1)
