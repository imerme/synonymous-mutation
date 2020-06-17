# amira
# 20200616
# rainy


class AppConfig(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.CUDA_VISIBLE_DEVICES = 0

    def set_params(self, key, value):
        self.kwargs[key] = value

    def get_params(self, key):
        return self.kwargs[key]

    def extent(self, d):
        self.kwargs.update(d)

    def __new__(cls, **kwargs):
        if not hasattr(cls, '_instance'):
            AppConfig._instance = super().__new__(cls)
        return AppConfig._instance
