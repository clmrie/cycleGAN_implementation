
import numpy as np
from dataset_loading import batch_generator
from dataset_loading import show_test_samples
from dataset_loading import save_model_weights

import os

def train(gen_AB, gen_BA, disc_A, disc_B, gan, patch_gan_shape, epochs, path, batch_size, img_shape, d_losses, g_losses, weights_path, viz_path):

	real_y = np.ones((batch_size, ) + patch_gan_shape)
	fake_y = np.zeros((batch_size, ) + patch_gan_shape)

	for epoch in range(epochs):

		print('Epoch {}'.format(epoch))

		for idx, (imgs_A, imgs_B) in enumerate(batch_generator(path, batch_size, img_shape = img_shape, is_testing = False)):

			fake_B = gen_AB.predict(imgs_A)
			fake_A = gen_BA.predict(imgs_B)

			disc_A_loss_real = disc_A.train_on_batch(imgs_A, real_y)
			disc_A_loss_fake = disc_B.train_on_batch(fake_A, fake_y)
			disc_A_loss = np.add(disc_A_loss_real, disc_A_loss_fake)/2

			disc_B_loss_real = disc_B.train_on_batch(imgs_B, real_y)
			disc_B_loss_fake = disc_B.train_on_batch(fake_B, fake_y)
			disc_B_loss = np.add(disc_B_loss_real, disc_B_loss_fake)/2

			disc_loss = np.add(disc_A_loss, disc_B_loss)/2

			gen_loss = gan.train_on_batch([imgs_A, imgs_B],
										  [real_y, real_y,
										  imgs_A, imgs_B,
										  imgs_A, imgs_B])

			d_losses.append(np.array([disc_A_loss, disc_B_loss, disc_loss]))
			g_losses.append(np.array(gen_loss))

		show_test_samples(path, 2, img_shape, gen_AB, gen_BA, viz_path, epoch)

		save_model_weights(weights_path, gen_AB, gen_BA, epoch)




