import cv2
import numpy as np

from attack import Attack
from watermark import Watermark


class DCT_Watermark(Watermark):
    def __init__(self):
        self.Q = 10
        self.size = 2
        self.sig_size = 150  # Increased from default 100 in Watermark class

    def inner_embed(self, B: np.ndarray, signature):
        sig_size = self.sig_size
        size = self.size

        w, h = B.shape[:2]
        
        # Calculate center position for the watermark
        center_x = (w - sig_size * size) // 2
        center_y = (h - sig_size * size) // 2
        
        # Ensure center_x and center_y are non-negative
        center_x = max(0, center_x)
        center_y = max(0, center_y)
        
        # Use only one position - the center of the image
        embed_pos = [(center_x, center_y)]

        for x, y in embed_pos:
            for i in range(x, x+sig_size * size, size):
                for j in range(y, y+sig_size*size, size):
                    # Make sure we don't go out of bounds
                    if i+size <= w and j+size <= h:
                        v = np.float32(B[i:i + size, j:j + size])
                        v = cv2.dct(v)
                        v[size-1, size-1] = self.Q * \
                            signature[((i-x)//size) * sig_size + (j-y)//size]
                        v = cv2.idct(v)
                        maximum = max(v.flatten())
                        minimum = min(v.flatten())
                        if maximum > 255:
                            v = v - (maximum - 255)
                        if minimum < 0:
                            v = v - minimum
                        B[i:i+size, j:j+size] = v
        return B

    def inner_extract(self, B):
        sig_size = self.sig_size
        size = self.size
        w, h = B.shape[:2]
        
        # Calculate the same center position used for embedding
        center_x = (w - sig_size * size) // 2
        center_y = (h - sig_size * size) // 2
        center_x = max(0, center_x)
        center_y = max(0, center_y)
        
        # Create a debug image to visualize extraction area
        debug_img = np.zeros_like(B)
        # Mark the extraction area with a white rectangle
        extraction_area_width = min(sig_size * size, w - center_x)
        extraction_area_height = min(sig_size * size, h - center_y)
        debug_img[center_y:center_y+extraction_area_height, center_x:center_x+extraction_area_width] = 255
        
        ext_sig = np.zeros(sig_size**2, dtype=np.int32)
        
        x, y = center_x, center_y
        for i in range(x, min(x+sig_size*size, w), size):
            for j in range(y, min(y+sig_size*size, h), size):
                if i+size <= w and j+size <= h:
                    v = cv2.dct(np.float32(B[i:i+size, j:j+size]))
                    if v[size-1, size-1] > self.Q / 2:
                        ext_sig[((i-x)//size) * sig_size + (j-y)//size] = 1
        return [ext_sig]


if __name__ == "__main__":
    img = cv2.imread("./images/cover.jpg")
    wm = cv2.imread("./images/watermark.jpg", cv2.IMREAD_GRAYSCALE)
    dct = DCT_Watermark()
    wmd = dct.embed(img, wm)
    cv2.imwrite("./images/watermarked.jpg", wmd)

    img = cv2.imread("./images/watermarked.jpg")

    img = Attack.gray(img)
    cv2.imwrite("./images/watermarked.jpg", img)

    dct = DCT_Watermark()
    signature = dct.extract(img)
    cv2.imwrite("./images/signature.jpg", signature)
