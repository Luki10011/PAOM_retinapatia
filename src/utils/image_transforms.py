
import numpy as np
from PIL import Image
import cv2

class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image):
        # Convert PIL image to numpy array
        img = np.array(img)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return Image.fromarray(img)
    
class CropTransform:
    def crop_image_from_gray(self, img, tol=7):
        if img.ndim == 2:
            mask = img > tol
            return img[np.ix_(mask.any(1), mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img > tol
            check_shape = img[:,:,0][np.ix_(mask.any(1), mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1 = img[:,:,0][np.ix_(mask.any(1), mask.any(0))]
                img2 = img[:,:,1][np.ix_(mask.any(1), mask.any(0))]
                img3 = img[:,:,2][np.ix_(mask.any(1), mask.any(0))]
                img = np.stack([img1, img2, img3], axis=-1)
            return img

    def __call__(self, img: Image.Image):
        img = np.array(img)
        img = self.crop_image_from_gray(img)
        img = Image.fromarray(img)

        return img
    

