import keras.models
from keras.preprocessing import image
from keras.layers import Input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import numpy as np

class Extractor():
    def __init__(self):
        self.weights = weights
        if weights is None:
            base_model = InceptionV3(weights='imagenet', include_top=True)
            self.model = keras.models.Model(
                inputs=base_model.input,
                outputs=base_models.get_layer('avg_pool').output
            )
        else:
            self.model = keras.models.load_model(weights)
            self.model.layers.pop()
            self.model.outputs = [self.model.layers[-1].output]
            self.model.output_layers = [self.model.layers[-1]]
            self.model.layers[-1].outbound_nodes = []
    
    def extract(self, image_path):
        img = image.load_img(image_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        features = self.model.predict(x)