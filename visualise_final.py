from tensorflow.python import pywrap_tensorflow
import numpy as np
from PIL import Image
checkpoint_path = "./model/model150.ckpt"
reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
var_to_shape_map = reader.get_variable_to_shape_map()
a=reader.get_tensor('conv2d/kernel')
print("tensor_name: ", var_to_shape_map)
tmp1=[]
for i in range(8):
	for j in range(8):
		if( j==0):
			tmp1.append(a[:,:,0,i*8+j])
		else:
			tmp1[i]=np.concatenate((tmp1[i],a[:,:,0,i*8+j]),axis=1)
				
tmp=np.concatenate((tmp1[0],tmp1[1],tmp1[2],tmp1[3],tmp1[4],tmp1[5],tmp1[6],tmp1[7]))
img = Image.fromarray(tmp, 'L')
img.save('my3.png')
image = Image.open('my3.png')
img.resize((100, 100),Image.ANTIALIAS)
img.save('my3.png')
#size = 10, 10
#img.thumbnail(size, Image.ANTIALIAS)
#img.save('my3_new.png', "JPEG")
img.show()
print(len(tmp1))

##print(reader.get_tensor('conv2d_2/kernel').shape)
#for key in var_to_shape_map:
#    print("tensor_name: ", key)
#    print(reader.get_tensor(key))