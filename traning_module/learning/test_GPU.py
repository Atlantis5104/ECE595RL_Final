#import tensorflow as tf
#print("GPUs Available: ", tf.config.list_physical_devices('GPU'))

import torch
print("CUDA available:", torch.cuda.is_available())

