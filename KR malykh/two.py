from skimage import io
import matplotlib.pyplot as plt
from skimage.transform import resize, rescale
from skimage.color import gray2rgb

url = "https://sotni.ru/wp-content/uploads/2023/08/natsionalnyi-park-banf-kanada-21.webp"
image = io.imread(url)

scaled_image = rescale(image, 0.5, anti_aliasing=True, channel_axis=-1)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.imshow(scaled_image)
ax.set_title("Scaled (50%)")
ax.axis("off")
plt.show()
