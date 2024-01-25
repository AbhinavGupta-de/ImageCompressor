# Image Compression using Singular Value Decomposition

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os


img = imread('./assests/redMountain.jpg')
imgGray = np.mean(img, -1)  # convert to grayscale

imgGray = imgGray/255  # normalize values
# plt.axis('off')
# plt.imshow(imgGray, cmap='gray')
# plt.show()


U, S, VT = np.linalg.svd(imgGray, full_matrices=False)
S = np.diag(S)


plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values')
plt.show()

plt.figure(2)
plt.plot(np.cumsum(np.diag(S))/np.sum(np.diag(S)))
plt.title('Cumulative Energy')
plt.show()


# j = 0
# for r in (5, 20, 100):
#     # Construct approximate image
#     imgApprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
#     plt.figure(j+1)
#     plt.title('r = %s' % r)
#     plt.imshow(imgApprox, cmap='gray')
#     plt.axis('off')
#     j += 1
#     plt.show()
