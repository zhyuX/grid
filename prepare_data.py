import tarfile
import os

def prepare_data(data_name=None):
    path_data = os.path.expanduser('~')+'/data_grid2op'  # 数据目录
    if data_name is None:
        return 0
    else:
        output_path = os.path.abspath(os.path.join(path_data, "{}.tar.bz2".format(data_name)))
        tar = tarfile.open(output_path, "r:bz2")
        tar.extractall(path_data)
        tar.close()

prepare_data('l2rpn_neurips_2020_track1_small')