	
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Concatenate, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras.models import Model

def downsample_block(incoming_layer, num_filters, kernel_size):

	downsample_layer = Conv2D(num_filters, kernel_size = kernel_size, strides = 2, padding = 'same')(incoming_layer)
	downsample_layer = LeakyReLU(alpha = 0.2)(downsample_layer)
	downsample_layer = InstanceNormalization()(downsample_layer)

	return downsample_layer


def upsample_block(incoming_layer, skip_input_layer, num_filters, kernel_size, dropout_rate = 0):

	upsample_layer = UpSampling2D(size = 2)(incoming_layer)
	upsample_layer = Conv2D(num_filters, kernel_size = kernel_size, strides = 1, padding = 'same', activation = 'relu')(upsample_layer)

	if dropout_rate: 
		upsample_layer = Dropout(dropout_rate)(upsample_layer)

	upsample_layer = InstanceNormalization()(upsample_layer)
	upsample_layer = Concatenate()([upsample_layer, skip_input_layer])

	return upsample_layer


def build_generator(img_shape, channels, num_filters):

	input_layer = Input(shape = img_shape)

	# Downsampling
	down_sample1 = downsample_block(input_layer, num_filters, kernel_size = 4)
	down_sample2 = downsample_block(down_sample1, num_filters * 2, kernel_size = 4)
	down_sample3 = downsample_block(down_sample2, num_filters * 4, kernel_size = 4)
	down_sample4 = downsample_block(down_sample3, num_filters * 8, kernel_size = 4)

	# Upsampling with skip connections
	up_sample1 = upsample_block(down_sample4, down_sample3, num_filters * 4, kernel_size = 4)
	up_sample2 = upsample_block(up_sample1, down_sample2, num_filters * 2, kernel_size = 4)
	up_sample3 = upsample_block(up_sample2, down_sample1, num_filters, kernel_size = 4)

	up_sample4 = UpSampling2D(size = 2)(up_sample3)
	output_img = Conv2D(channels, kernel_size = 4, strides = 1, padding = 'same', activation = 'tanh')(up_sample4)

	return Model(input_layer, output_img)





