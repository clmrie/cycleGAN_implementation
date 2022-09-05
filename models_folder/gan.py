

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout, Concatenate, LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

def build_gan(input_shape, gen_AB, gen_BA, disc_A, disc_B, lambda_cycle, lambda_identity):

	img_A = Input(shape = input_shape)
	img_B = Input(shape = input_shape)

	# Fake samples
	fake_B = gen_AB(img_A)
	fake_A = gen_BA(img_B)

	# Reconstructing original samples
	reconstruct_A = gen_BA(fake_B)
	reconstruct_B = gen_AB(fake_A)

	# Identity samples
	identity_A = gen_BA(img_A)
	identity_B = gen_AB(img_B)

	# Disabling discriminator training (only updating the generators' weights)
	disc_A.trainable = False
	disc_B.trainable = False

	valid_A  = disc_A(fake_A)
	valid_B = disc_B(fake_B)

	gan = Model(inputs = [img_A, img_B],
				outputs = [valid_A, valid_B,
						   reconstruct_A, reconstruct_B,
						   identity_A, identity_B])

	gan.compile(loss = ['mse', 'mse',
						'mae', 'mae',
						'mae', 'mae'],
				loss_weights = [1, 1,
								lambda_cycle, lambda_cycle,
								lambda_identity, lambda_identity],
				optimizer = Adam(0.0002, 0.5))

	return gan





