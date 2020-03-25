class Path(object):
    @staticmethod
    def db_dir(database):
        root_dir = './aps_cut'
        output_dir = './data'
        return root_dir, output_dir

    @staticmethod
    def model_dir():
        return './c3d-pretrained.pth'