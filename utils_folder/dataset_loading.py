
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from glob import glob

def imread(path, img_shape):

	img = plt.imread(path, format = 'RGB').astype(np.float)
	img = tf.image.resize(img, img_shape).numpy()

	img = img / 127.5 - 1

	return img

def show_test_samples(path, nb_samples, img_shape, gen_AB, gen_BA, saving_path, epoch):

	data_type = 'test'
	paths_A = glob(os.path.join(path, '{}A'.format(data_type), '*'))
	paths_B = glob(os.path.join(path, '{}B'.format(data_type), '*'))

	paths_A = np.random.choice(paths_A, nb_samples, replace = False)
	paths_B = np.random.choice(paths_B, nb_samples, replace = False)

	saving_path = os.path.join(saving_path, str(epoch))
	os.mkdir(saving_path)

	for idx, (img_A, img_B) in enumerate(zip(paths_A, paths_B)):

		img_A = (imread(img_A, img_shape)+1)/2
		img_B = (imread(img_B, img_shape)+1)/2

		img_A = np.array([img_A])
		img_B = np.array([img_B])

		fake_B = (gen_AB.predict(img_A)+1)/2
		fake_A = (gen_BA.predict(img_B)+1)/2 

		print('Test samples', idx+1)
		print('\n-------------')

		print('ori_A')
		img_A = np.squeeze(img_A)
		plt.imshow(img_A)
		plt.imsave(os.path.join(saving_path, 'ori_A_{}.jpg'.format(idx)), img_A)
		plt.show()

		print('fake_B')
		fake_B = np.squeeze(fake_B)
		plt.imshow(fake_B)
		plt.imsave(os.path.join(saving_path, 'fake_B{}.jpg'.format(idx)), fake_B)
		plt.show()

		print('ori_B')
		img_B = np.squeeze(img_B)
		plt.imshow(img_B)
		plt.imsave(os.path.join(saving_path, 'ori_B{}.jpg'.format(idx)), img_B)
		plt.show()

		print('fake_A')
		fake_A = np.squeeze(fake_A)
		plt.imshow(fake_A)
		plt.imsave(os.path.join(saving_path, 'fake_A{}.jpg'.format(idx)), fake_A)
		plt.show()

def save_model_weights(saving_path, gen_AB, gen_BA, epoch):

	weights_path = os.path.join(saving_path, str(epoch))
	os.mkdir(weights_path)

	gen_AB.save(os.path.join(weights_path, 'gen_AB.h5'))
	gen_BA.save(os.path.join(weights_path, 'gen_BA.h5'))


def batch_generator(path, batch_size, img_shape, is_testing):

	data_type = 'train' if not is_testing else 'test'
	paths_A = glob(os.path.join(path, '{}A'.format(data_type), '*'))
	paths_B = glob(os.path.join(path, '{}B'.format(data_type), '*'))
    
	nb_batches = int(min(len(paths_A), len(paths_B)) / batch_size)
	nb_samples = nb_batches * batch_size

	paths_A = np.random.choice(paths_A, nb_samples, replace = False)
	paths_B = np.random.choice(paths_B, nb_samples, replace = False)
    
	for i in range(nb_batches):

		batch_A = paths_A[i * batch_size: (i+1) * batch_size]
		batch_B = paths_B[i * batch_size: (i+1) * batch_size]

		imgs_A, imgs_B = [], []
		for img_A, img_B in zip(batch_A, batch_B):

			img_A = imread(img_A, img_shape)
			img_B = imread(img_B, img_shape)

			if not is_testing and np.random.random() > 0.5:

				img_A = np.fliplr(img_A)
				img_B = np.fliplr(img_B)

			imgs_A.append(img_A)
			imgs_B.append(img_B)

		imgs_A = np.array(imgs_A)
		imgs_B = np.array(imgs_B)

		yield imgs_A, imgs_B


