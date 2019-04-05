"""RC Car Dataset."""

from torch.utils import data

# import cv2
import numpy as np

# use skimage if you do not have cv2 installed
from skimage import io


def augment(dataroot, imgName, angle):
    """Data augmentation."""
    name = dataroot + 'IMG/' + imgName.split('\\')[-1]
    # current_image = cv2.imread(name)
    current_image = io.imread(name)

    current_image = current_image[65:-25, :, :]
    if np.random.rand() < 0.5:
        # current_image = cv2.flip(current_image, 1)
        current_image = np.flipud(current_image)
        angle = angle * -1.0

    return current_image, angle


class TripletDataset(data.Dataset):

    def __init__(self, dataroot, samples, transform=None):
        self.samples = samples
        self.dataroot = dataroot
        self.transform = transform

    def __getitem__(self, index):
        batch_samples  = self.samples[index]
        steering_angle = float(batch_samples[3])

        center_img, steering_angle_center = augment(self.dataroot, batch_samples[0], steering_angle)
        left_img, steering_angle_left     = augment(self.dataroot, batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right   = augment(self.dataroot, batch_samples[2], steering_angle - 0.4)

        center_img = self.transform(center_img)
        left_img   = self.transform(left_img)
        right_img  = self.transform(right_img)

        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)

    def __len__(self):
        return len(self.samples)
