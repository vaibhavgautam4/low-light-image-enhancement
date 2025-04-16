# enhance.py
from PIL import Image
import numpy as np
import tensorflow as tf
from models.LIEnet import lienet

def closest_number(n, m):
    return int(n - (n % m))

class Inferer:
    def __init__(self):
        self.model = None

    def build_model(self, num_rrg: int, num_mrb: int, channels: int, weights_path: str):
        self.model = lienet(
            image_size=None, num_rrg=num_rrg,
            num_mrb=num_mrb, channels=channels
        )
        self.model.load_weights(weights_path)

    def _predict(self, original_image, image_resize_factor: float = 1.):
        width, height = original_image.size
        target_width, target_height = (
            closest_number(width // image_resize_factor, 4),
            closest_number(height // image_resize_factor, 4)
        )
        original_image = original_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        image = tf.keras.preprocessing.image.img_to_array(original_image)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        output = self.model.predict(image)
        output_image = output[0] * 255.0
        output_image = output_image.clip(0, 255)
        output_image = output_image.reshape(
            (np.shape(output_image)[0], np.shape(output_image)[1], 3)
        )
        output_image = Image.fromarray(np.uint8(output_image))
        return output_image

    def infer(self, image_path, image_resize_factor: float = 1.):
        original_image = Image.open(image_path)
        output_image = self._predict(original_image, image_resize_factor)
        return original_image, output_image

    def infer_streamlit(self, image_pil, image_resize_factor: float = 1.):
        output_image = self._predict(image_pil, image_resize_factor)
        return image_pil, output_image
