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
from encoder.generator_model import Generator
from encoder.perceptual_model import PerceptualModel
import runway



@runway.setup(options={'people_vector': runway.file(extension='.npy'), 'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
	# load latent representation
	global latent_vector
	p1 = opts['people_vector']
	latent_vector = np.load(p1)
	tflib.init_tf()
	model = opts['checkpoint']
	print("open model %s" % model)
	with open(model, 'rb') as file:
		G, D, Gs = pickle.load(file)
	Gs.print_layers()
	global generator
	generator = Generator(Gs, batch_size=1, randomize_noise=False)
	return Gs

# def setup(opts):
# 	tflib.init_tf()
# 	url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
# 	with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
# 		_G, _D, Gs = pickle.load(f)
# 		# _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
# 		# _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
# 		# Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.
# 	Gs.print_layers()
# 	global generator
# 	generator = Generator(Gs, batch_size=1, randomize_noise=False)
# 	return Gs

def generate_image(generator, latent_vector):
	latent_vector = latent_vector.reshape((1, 18, 512))
	generator.set_dlatents(latent_vector)
	img_array = generator.generate_images()[0]
	img = PIL.Image.fromarray(img_array, 'RGB')
	return img.resize((512, 512))   

generate_inputs = {
	'age': runway.number(min=-26, max=26, default=6, step=0.1)
}

generate_outputs = {
	'image': runway.image(width=512, height=512)
}

@runway.command('generat3r', inputs=generate_inputs, outputs=generate_outputs)
def move_and_show(model, inputs):
	# load direction
	age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
	direction = age_direction
	# generator
	coeff = inputs['age']
	new_latent_vector = latent_vector.copy()
	new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
	image = (generate_image(generator, new_latent_vector))
	#ax[i].set_title('Coeff: %0.1f' % coeff)
	#plt.show()
	return {'image': image}

if __name__ == '__main__':
	runway.run(debug=True)