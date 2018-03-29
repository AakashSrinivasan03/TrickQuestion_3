from tensorflow.python import pywrap_tensorflow
import numpy as np
from PIL import Image
checkpoint_path = "./model/model150.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
a=reader.get_tensor('conv2d/kernel')
print("tensor_name: ", var_to_shape_map)
img = Image.fromarray(a[:,:,0,25], 'L')
img.save('my_25.png')
img.show()