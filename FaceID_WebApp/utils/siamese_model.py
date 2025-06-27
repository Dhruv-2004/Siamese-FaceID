import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import os

class L1Dist(Layer):
    """Custom L1 Distance layer for Siamese network"""
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        input_embedding, validation_embedding = inputs
        return tf.math.abs(input_embedding - validation_embedding)

def make_embeddings():
    """Create the embedding network architecture"""
    inp = Input(shape=(105, 105, 3), name="input_image")

    # First Block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # Second Block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # Third Block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # Fourth Block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)

    return Model(inputs=inp, outputs=d1, name='embedding')

def make_siamese_model():
    """Create the complete Siamese network"""
    # Create embedding network
    embedding = make_embeddings()
    
    # Define inputs
    input_image = Input(name='input_img', shape=(105, 105, 3))
    validation_image = Input(name='validation_img', shape=(105, 105, 3))
    
    # Combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer([embedding(input_image), embedding(validation_image)])

    # Classification layer 
    classifier = Dense(1, activation='sigmoid')(distances)
    
    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

def load_siamese_model(model_path):
    """Load the trained Siamese model with custom objects"""
    # Define custom objects for loading
    custom_objects = {'L1Dist': L1Dist}
    
    try:
        # Load the model
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # If loading fails, create new model and load weights
        model = make_siamese_model()
        try:
            model.load_weights(model_path.replace('.h5', '_weights.h5'))
        except:
            # Try loading from checkpoint
            checkpoint_path = model_path.replace('siamese_network.h5', '../training_checkpoints/siamese_ckpt.weights.h5')
            if os.path.exists(checkpoint_path):
                model.load_weights(checkpoint_path)
            else:
                print("Warning: No trained weights found. Using untrained model.")
        return model

def preprocess_image(image_path):
    """Preprocess image for model input"""
    try:
        # Read and preprocess the image
        img = tf.io.read_file(image_path)
        img = tf.io.decode_image(img, channels=3)
        img = tf.image.resize(img, [105, 105])
        img = tf.cast(img, tf.float32) / 255.0
        return img
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

def preprocess_image_array(image_array):
    """Preprocess numpy image array for model input"""
    try:
        # Resize and normalize
        img = tf.image.resize(image_array, [105, 105])
        img = tf.cast(img, tf.float32) / 255.0
        return img
    except Exception as e:
        print(f"Error preprocessing image array: {e}")
        return None 