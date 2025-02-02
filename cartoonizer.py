import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Set figure size for plotting
plt.rcParams['figure.figsize'] = (12, 6)

# Load the image
img = cv2.imread('therock.png')
if img is None:
    raise FileNotFoundError("Error: 'therock.png' not found. Please check the file path.")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display the original image
plt.axis("off")
plt.imshow(img)
plt.title("Original Image")
plt.show()

# Edge mask generation
line_size = 7
blur_value = 7

gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
gray_blur = cv2.medianBlur(gray_img, blur_value)
edges = cv2.adaptiveThreshold(
    gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value
)

# Display the edge mask
plt.axis("off")
plt.imshow(edges, cmap='gray')
plt.title("Edge Mask")
plt.show()

# Color quantization using KMeans clustering
k = 7
data = img.reshape(-1, 3)

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
img_reduced = kmeans.cluster_centers_[kmeans.labels_]
img_reduced = img_reduced.reshape(img.shape).astype(np.uint8)

# Display the reduced color image
plt.axis("off")
plt.imshow(img_reduced)
plt.title("Color Quantized Image")
plt.show()

# Bilateral filter to smooth colors
blurred = cv2.bilateralFilter(img_reduced, d=7, sigmaColor=200, sigmaSpace=200)

# Combine edges and blurred image to create the cartoon effect
cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

# Display the original vs. cartoonized image
plt.subplot(1, 2, 1)
plt.axis("off")
plt.imshow(img)
plt.title("Original")

plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(cartoon)
plt.title("Cartoonized")

plt.show()

# Export cartoon image to a file
cartoon_ = cv2.cvtColor(cartoon, cv2.COLOR_RGB2BGR)
cv2.imwrite('cartoon.png', cartoon_)

print("Cartoon image saved as 'cartoon.png'")