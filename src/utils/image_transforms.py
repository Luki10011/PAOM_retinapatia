
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
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        mask = gray > 30
        coords = np.argwhere(mask)
        y0, x0 = coords.min(axis = 0)
        y1, x1 = coords.max(axis = 0)
        cropped = img[y0:y1, x0:x1]
        return cropped

    def __call__(self, img: Image.Image):
        img = np.array(img)
        img = self.crop_image_from_gray(img)
        return Image.fromarray(img)
    
class ResizeTransform:
    def smart_resize(img, size=(1024,1024)):
        h, w = img.shape[:2]
        if h > size[0] and w > size[1]:
            interp = cv2.INTER_AREA
        elif h < size[0] and w < size[1]:
            interp = cv2.INTER_CUBIC
        else:
            interp = cv2.INTER_LINEAR
        return cv2.resize(img, size, interpolation = interp)
