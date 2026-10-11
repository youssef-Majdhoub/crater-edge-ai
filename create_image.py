import numpy as np
import matplotlib.pyplot as plt

image = np.zeros((200, 200), dtype=np.uint8)
image[0:100, 0:100] = np.random.randint(0, 64, (100, 100), dtype=np.uint8)
image[0:100, 100:200] = np.random.randint(64, 128, (100, 100), dtype=np.uint8)
image[100:200, 100:200] = np.random.randint(128, 192, (100, 100), dtype=np.uint8)
image[100:200, 0:100] = np.random.randint(192, 256, (100, 100), dtype=np.uint8)
# show the image
plt.imshow(image, cmap="gray")
plt.show()
np.save("image.npy", image)
