# https://colab.research.google.com/drive/1RUaVUqCvyojwoMglp6cFoLDnCfLHBZtB#scrollTo=PVYrjvgE_8AU

import helpers
import os
import argparse
import pickle
from tqdm import tqdm
import PIL.Image
import numpy as np
import dnnlib
import dnnlib.tflib as tflib
import config
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import runway

prevIterations = -1


def load_local_image(images_path, img_size):
	img = image.load_img(images_path, target_size=(img_size, img_size))
	img = image.img_to_array(img)
	img = np.expand_dims(img, 0)
	#_img.resize(img_size, img_size)  
	loaded_image = np.vstack(img)
	preprocessed_image = preprocess_input(loaded_image)
	return preprocessed_image


@runway.setup
def setup(opts):
	tflib.init_tf()
	url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
	with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
		_G, _D, Gs = pickle.load(f)
		# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
		# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
		# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
	Gs.print_layers()
	global generator
	#generator = Generator(Gs, batch_size=1, randomize_noise=False)
	global perceptual_model
	perceptual_model = PerceptualModel(256, layer=9, batch_size=1)
	#perceptual_model.build_perceptual_model(generator.generated_image)
	return Gs

# @runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
# def setup(opts):
# 	# Initialize generator and perceptual model
# 	tflib.init_tf()
# 	model = opts['checkpoint']
# 	print("open model %s" % model)
# 	with open(model, 'rb') as file:
# 		G, D, Gs = pickle.load(file)
# 	Gs.print_layers()
# 	global generator
# 	generator = Generator(Gs, batch_size=1, randomize_noise=False)
# 	perceptual_model = PerceptualModel(256, layer=9, batch_size=1)
# 	perceptual_model.build_perceptual_model(generator.generated_image)
# 	return Gs


generate_inputs = {
	'portrait': runway.image(),
	'iterations': runway.number(min=1, max=5000, default=10, step=1.0),
	'age': runway.number(min=-30, max=20, default=4, step=0.2)
}
# generate_outputs = {
# 	'latent_vector': runway.file()
# }

generate_outputs = {
#	'generated': runway.image(width=512, height=512)
	# 'vector': runway.vector(length=512)
	"hextext": runway.text
}

@runway.command('encode', inputs=generate_inputs, outputs=generate_outputs)
def find_in_space(model, inputs):
	#names = os.path.splitext(os.path.basename(inputs['portrait']))
	#print(names)
	global prevIterations
	if (inputs['iterations'] != prevIterations):
		# generator.reset_dlatents()
		names = ["looking at you!"]
		img = load_local_image("images/hec.jpg", 512)
		perceptual_model.set_reference_images(img)
		prevIterations = inputs['iterations']
		print ("encoding for: ", prevIterations)
		op = perceptual_model.optimize(generator.dlatent_variable, iterations=inputs[iterations], learning_rate=1.)
		pbar = tqdm(op, leave=False, total=inputs[iterations])
		for loss in pbar:
			pbar.set_description(' '.join(names)+' Loss: %.2f' % loss)
		print(' '.join(names), ' loss:', loss)		


	# Generate images from found dlatents and save them
	generated_images = generator.generate_images()
	generated_dlatents = generator.get_dlatents()
	for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
		img = PIL.Image.fromarray(img_array, 'RGB')
		img.resize((512, 512))  
	#	img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
	#	np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

	#return {'latent_vector' : dlatent}
	# return {'image': img}
	print ("returning text")
	#return {'vector': generated_dlatents}
	#return {'generated': img}
	s = generated_dlatents.tostring()
	s.hex()
	return{"hextext": s}

if __name__ == '__main__':
	runway.run(debug=True)