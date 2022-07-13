import pims_nd2 as nd2
import os
import cv2
import numpy as np


def read_images_from_dir(dir):
   """
   Reads in all images from specified directly and returns a list of numpy arrays

   Params:
      dir: (str) path of the directory containing images to read
    
   Returns:
      images: ([np.array]) a list of images as np.arrays
   """

   images = []

   for i,filename in enumerate(os.listdir(dir)):
      images.append(cv2.imread(os.path.join(dir,filename)))
   
   return images


def detect_droplets(img): 
    """
    Detects all of the droplets in the input image using the connected component
    alg and returns a list of droplets

    Params:
      img: (np.array) input image for now 2D
    
    Returns:
      dims: (np.array of size (number droplets, 4)) matrix of width, height, left, and top of 
      bounding box for each droplet
      centroids: (np.array of size(number droplets, 2)) matrix of x,y coords of 
      the centroid for each droplet
    """
    tmp = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY_INV,11,21)   

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(np.array(tmp), connectivity=8)

    #do not include the first row of the stats matrix, it is just the shape of the image
    width = np.expand_dims(stats[1:, cv2.CC_STAT_WIDTH],axis = -1)
    height = np.expand_dims(stats[1:, cv2.CC_STAT_HEIGHT], axis = - 1)
    left = np.expand_dims(stats[1:, cv2.CC_STAT_LEFT],axis = -1)
    top = np.expand_dims(stats[1:, cv2.CC_STAT_TOP],axis = -1)
    

    #combine width and height
    dims = np.concatenate((width,height,left,top),axis = 1)

    return dims, centroids


def crop_droplets(img,dims,startCount,outhPath = '',save = False):
  
  droplets = []
  count = startCount
  
  for i in range(dims.shape[0]):
      w = dims[i][0]
      h = dims[i][1] 

      left = dims[i][2]
      top = dims[i][3] 
      crop = img[top: top + h +1,left :left + w + 1]

      if save:
        cv2.imwrite(outhPath + f'droplet_{count}.jpg',crop)
      
      #cv2_imshow(crop)
      droplets.append(crop)
      count += 1

  return droplets