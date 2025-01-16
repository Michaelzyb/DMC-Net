import numpy as np
import random
import cv2

normal_path = ''
defect_path = ''
defect_mask_path = ''
normal_img = cv2.imread(normal_path)
defect_img = cv2.imread(defect_path)
defect_label = cv2.imread(defect_mask_path, 0)
ratio = random.random() * 0.75 + 0.25
d_img = cv2.resize(defect_img, (0, 0), fx=ratio, fy=ratio)
d_label = cv2.resize(defect_label, (0, 0), fx=ratio, fy=ratio)
d_h, d_w, _ = d_img.shape
h, w, _ = normal_img.shape
# Select the upper-left point where the defective image will be placed.
pos_x = random.randrange(0, w-d_w, step=1)
pos_y = random.randrange(0, h-d_h, step=1)
# Creating overlaid labels and images
new_mask = np.zeros((h, w))
new_mask[pos_y:pos_y+d_h, pos_x:pos_x+d_w] = d_label
# Placement of defective images using superimposed labels
normal_img[new_mask.astype(np.bool_)] = d_img[d_label.astype(np.bool_)]

koutu_defect = np.zeros((defect_label.shape[0], defect_label.shape[1], 4), dtype=np.uint8)
koutu_defect[:, :, :3] = defect_img
koutu_defect[defect_label.astype(np.bool_), 3] = 255
cv2.imwrite('micro_defect.png', koutu_defect)
cv2.imwrite('new_defect_img.png', normal_img)