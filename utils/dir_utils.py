
import os

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        print('Made dir : {}'.format(path))
    else:
        print('Dir {} already exists'.format(path))