import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, models, losses, metrics, callbacks
import tensorflow_addons as tfa

# https://colab.research.google.com/drive/16Ft9geWhFyHtFeIkBlrxVt7o8-4Jx7ou#scrollTo=kkyY2FqK856-

# Prepare the data
num_classes = 100
input_shape = (32, 32, 3)

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()

print(f"x_train shape: {x_train.shape} - y_train shape: {y_train.shape}")
print(f"x_test shape: {x_test.shape} - y_test shape: {y_test.shape}")
# end


# Configure the hyperparameters
learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100

image_size = 72 # We'll resize input images to this size
patch_size = 6 # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2

projection_dim = 64
num_heads = 4

transformer_units = [
    projection_dim * 2,
    projection_dim
] # Size of the transformer layers
transformer_layers = 8

mlp_head_units = [2048, 1024] # Size of the dense layers of the final classifier
# end


# Use data augmentation
data_augmentation = Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(height_factor=0.2, width_factor=0.2)
    ],
    name='data_augmentation'
)

# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)
# end


# Implement multilayer perceptron (MLP)
def mlp(x, hidden_units, drop_rate):
    for unit in hidden_units:
        x = layers.Dense(unit)(x)
        x = layers.Activation(activation='gelu')(x)
        x = layers.Dropout(drop_rate)(x)

    return x
# end


# Implement patch creation as a layer
class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]

        # shape: [batch, 12, 12, 108]
        # [batch, num_patches, num_patches, patch_size*patch_size*image_dims]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        patch_dims = patches.shape[-1]

        patches = tf.reshape(patches, shape=[batch_size, -1, patch_dims])

        return patches
# end


# test Patches
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(4, 4))
# image = x_train[np.random.choice(range(x_train.shape[0]))]
# plt.imshow(image.astype("uint8"))
# plt.axis("off")
#
# resized_image = tf.image.resize(
#     tf.convert_to_tensor([image]), size=(image_size, image_size)
# )
# patches = Patches(patch_size)(resized_image)
# print(patches.shape)
# print(f"Image size: {image_size} X {image_size}")
# print(f"Patch size: {patch_size} X {patch_size}")
# print(f"Patches per image: {patches.shape[1]}")
# print(f"Elements per patch: {patches.shape[-1]}")
#
# n = int(np.sqrt(patches.shape[1]))
# plt.figure(figsize=(4, 4))
# for i, patch in enumerate(patches[0]):
#     print(patch.shape)
#     ax = plt.subplot(n, n, i + 1)
#     patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
#     plt.imshow(patch_img.numpy().astype("uint8"))
#     plt.axis("off")
#
# plt.show()
# end


# Implement the patch encoding layer
class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dims):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches

        self.projection = layers.Dense(projection_dims)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches,
            output_dim=projection_dims
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
# end


# Build the ViT model
# vit https://pic4.zhimg.com/80/v2-5afd38bd10b279f3a572b13cda399233_1440w.webp
def create_vit_classifier():
    inputs = layers.Input(shape=input_shape)
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=projection_dim,
            dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])

        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, drop_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, drop_rate=0.5)

    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    model = models.Model(inputs=inputs, outputs=logits)

    return model
# end


# Compile, train, and evaluate the mode
def run_experiment(model: models.Model):
    optimizer = tfa.optimizers.AdamW(
        weight_decay=weight_decay,
        learning_rate=learning_rate
    )

    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            metrics.SparseCategoricalAccuracy("accuracy"),
            metrics.SparseTopKCategoricalAccuracy(5, name='top-5-accuracy')
        ]
    )

    cpkt_path = './models/ckpt'
    ckpt_callback = callbacks.ModelCheckpoint(
        filepath=cpkt_path,
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_split=0.1,
        callbacks=[ckpt_callback]
    )

    model.load_weights(cpkt_path)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)

    print(f"Test accuracy: {round(accuracy * 100, 2)}%")
    print(f"Test top 5 accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


vit_model = create_vit_classifier()
history = run_experiment(vit_model)
# end
