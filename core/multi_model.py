import tensorflow as tf
from tensorflow import keras


class MultiModels(object):
    def __init__(self, sub_models, data):
        self.sub_model = sub_models

    def add_model(self, model, data):
