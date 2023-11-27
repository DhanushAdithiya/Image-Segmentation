import sys
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans


image = Image.open(sys.argv[1])
img_arr = np.array(image)
dis_img = Image.fromarray(img_arr, "RGB")


vec = img_arr.reshape((-1,3))
print(img_arr.shape)
print(vec.shape)


kmeans = KMeans(n_clusters = int(sys.argv[2]), random_state = 0, n_init = 1).fit(vec)
colors = np.uint8(kmeans.cluster_centers_)
seg = colors[kmeans.labels_.flatten()]


image_quant = seg.reshape(img_arr.shape)
quant = Image.fromarray(image_quant, "RGB")


quant.show()


