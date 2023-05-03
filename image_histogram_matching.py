import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_histogram(image):
	
	# 1st channel values histogram
	hist1 = np.zeros(256)
	
	for row in range(0, image.shape[0]):
		for col in range(0, image.shape[1]):
			pixel = image[row, col, 0]
			hist1[pixel] = hist1[pixel] + 1

	plt.hist(image[:,:,0])
	plt.show()
	plt.plot(hist1)
	
	print(hist1.max())
	
	plt.show()
	
	# 2nd channel values histogram
	hist2 = np.zeros(256)
	
	for row in range(0, image.shape[0]):
		for col in range(0, image.shape[1]):
			pixel = image[row, col, 1]
			hist2[pixel] = hist2[pixel] + 1

	plt.hist(image[:,:,1])
	plt.show()
	plt.plot(hist2)
	print(hist2.max())
	
	plt.show()
	
	# 3rd channel value histogram
	hist3 = np.zeros(256)
	
	for row in range(0, image.shape[0]):
		for col in range(0, image.shape[1]):
			pixel = image[row, col, 2]
			hist3[pixel] = hist3[pixel] + 1
				
	plt.hist(image[:,:,2])
	plt.show()
	plt.plot(hist3)
	print(hist3.max())
	
	
	plt.show()
	
	histogram = np.zeros(256*3)
	
	for i in range(0, 256):
		histogram[i] = hist1[i]
	
	for j in range(256, 256*2):
		histogram[j] = hist2[j - 256]
	
	for k in range(256*2, 256*3):
		histogram[k] = hist3[k-256*2]
	
	
	return histogram
	
def match_histogram(image1, histogram1, histogram2):

	cdf1 = np.zeros(256)
	cdf2 = np.zeros(256)
	cdf1[0] = histogram1[0]
	cdf2[0] = histogram2[0]
	for i in range(1, 256):
		cdf1[i] = cdf1[i-1] + histogram1[i]
	for j in range(1, 256):
		cdf2[j] = cdf2[j-1] + histogram2[j]
		
	
	plt.plot(cdf1)
	print(cdf1.max())	
	
	plt.show()

	plt.plot(cdf2)
	print(cdf2.max())
	
	plt.show()



image1 = cv2.imread("images.jpeg")
image2 = cv2.imread("images1.jpeg")

histogram1 = generate_histogram(image1)
histogram2 = generate_histogram(image2)



modified_image = match_histogram(image1, histogram1, histogram2)

#cv2.imshow("modified image", modified_image)
#cv2.imshow("source image", image1)
#cv2.imshow("reference image", image2)

#cv2.waitKey(0)
###

