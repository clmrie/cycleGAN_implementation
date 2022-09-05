
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Concatenate, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

def discriminator_block(incoming_layer, num_filters, kernel_size, instance_normalization):

	disc_layer = Conv2D(num_filters, kernel_size = kernel_size, strides = 2, padding = 'same')(incoming_layer)
	disc_layer = LeakyReLU(alpha = 0.2)(disc_layer)

	if instance_normalization:
		disc_layer = InstanceNormalization()(disc_layer)

	return disc_layer


def build_discriminator(img_shape, num_filters):

	input_layer = Input(shape = img_shape)

	disc_block1 = discriminator_block(input_layer, num_filters, kernel_size = 4, instance_normalization = False)
	disc_block2 = discriminator_block(disc_block1, num_filters * 2, kernel_size = 4, instance_normalization = True)
	disc_block3 = discriminator_block(disc_block2, num_filters * 4, kernel_size = 4, instance_normalization = True)
	disc_block4 = discriminator_block(disc_block3, num_filters * 8, kernel_size = 4, instance_normalization = True)

	output = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(disc_block4)

	discriminator = Model(input_layer, output)

	discriminator.compile(loss = 'mse', optimizer = Adam(0.0002, 0.5), metrics = ['accuracy'])

	return discriminator