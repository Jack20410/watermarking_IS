import cv2
import numpy as np
import math

class Attack:
    @staticmethod
    def blur(img: np.ndarray):
        return cv2.blur(img, (2, 2))

    @staticmethod
    def rotate180(img: np.ndarray):
        img = img.copy()
        angle = 180
        scale = 1.0
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array(
            [(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    @staticmethod
    def rotate90(img: np.ndarray):
        img = img.copy()
        angle = 90
        scale = 1.0
        w = img.shape[1]
        h = img.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        nw = (abs(np.sin(rangle) * h) + abs(np.cos(rangle) * w)) * scale
        nh = (abs(np.cos(rangle) * h) + abs(np.sin(rangle) * w)) * scale
        rot_mat = cv2.getRotationMatrix2D((nw * 0.5, nh * 0.5), angle, scale)
        rot_move = np.dot(rot_mat, np.array(
            [(nw - w) * 0.5, (nh - h) * 0.5, 0]))
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        return cv2.warpAffine(img, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))), flags=cv2.INTER_LANCZOS4)

    @staticmethod
    def chop5(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        return img[int(w * 0.05):, :]

    @staticmethod
    def chop10(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        return img[int(w * 0.1):, :]

    @staticmethod
    def chop30(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        return img[int(w * 0.3):, :]

    @staticmethod
    def gray(img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray

    @staticmethod
    def saltnoise(img: np.ndarray):
        img = img.copy()
        for k in range(1000):
            i = int(np.random.random() * img.shape[1])
            j = int(np.random.random() * img.shape[0])
            if img.ndim == 2:
                img[j, i] = 255
            elif img.ndim == 3:
                img[j, i, 0] = 255
                img[j, i, 1] = 255
                img[j, i, 2] = 255
        return img

    @staticmethod
    def randline(img: np.ndarray):
        img = img.copy()
        h, w = img.shape[:2]
        
        # Calculate center region coordinates
        center_x = w // 2
        center_y = h // 2
        rect_width = w // 3
        rect_height = h // 3
        
        # Draw rectangles in the center region
        cv2.rectangle(img, 
                     (center_x - rect_width//2, center_y - rect_height//2), 
                     (center_x + rect_width//2, center_y + rect_height//2), 
                     (0, 255, 0), 3)
        
        cv2.rectangle(img, 
                     (center_x - rect_width//4, center_y - rect_height//4), 
                     (center_x + rect_width//4, center_y + rect_height//4), 
                     (255, 0, 0), 3)
        
        # Draw crossing lines through the center of the image
        cv2.line(img, (center_x - rect_width//2, center_y - rect_height//2), 
                (center_x + rect_width//2, center_y + rect_height//2), 
                (255, 0, 0), 5)
        
        cv2.line(img, (center_x - rect_width//2, center_y + rect_height//2), 
                (center_x + rect_width//2, center_y - rect_height//2), 
                (255, 0, 255), 5)
        
        return img

    @staticmethod
    def cover(img: np.ndarray):
        img = img.copy()
        # Get image dimensions
        h, w = img.shape[:2]
        # Place circle in the middle of the image
        center_x = w // 2
        center_y = h // 2
        # Calculate radius as 20% of the smallest dimension
        radius = min(w, h) // 5
        # Draw filled circle in the center
        cv2.circle(img, (center_x, center_y), radius, (0, 0, 255), -1)  # Red circle
        # Add a white border to make it more visible
        cv2.circle(img, (center_x, center_y), radius, (255, 255, 255), 3)  # White border
        
        # Add text at the bottom of the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = w / 500  # Scale font based on image width
        cv2.putText(img, 'Image Altered', (w//4, h-50), font, font_scale, (255, 255, 0), 2)
        return img

    @staticmethod
    def brighter10(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        for xi in range(0, w):
            for xj in range(0, h):
                img[xi, xj, 0] = int(img[xi, xj, 0] * 1.1)
                img[xi, xj, 1] = int(img[xi, xj, 1] * 1.1)
                img[xi, xj, 2] = int(img[xi, xj, 2] * 1.1)
        return img

    @staticmethod
    def darker10(img: np.ndarray):
        img = img.copy()
        w, h = img.shape[:2]
        for xi in range(0, w):
            for xj in range(0, h):
                img[xi, xj, 0] = int(img[xi, xj, 0] * 0.9)
                img[xi, xj, 1] = int(img[xi, xj, 1] * 0.9)
                img[xi, xj, 2] = int(img[xi, xj, 2] * 0.9)
        return img

    @staticmethod
    def largersize(img: np.ndarray):
        w, h = img.shape[:2]
        return cv2.resize(img, (int(h * 1.5), w))

    @staticmethod
    def smallersize(img: np.ndarray):
        w, h = img.shape[:2]
        return cv2.resize(img, (int(h * 0.5), w))

if __name__ == "__main__":
    img = cv2.imread("./images/watermarked.jpg")
    img = Attack.blur(img)
    cv2.imwrite("./images/watermarked.jpg", img)