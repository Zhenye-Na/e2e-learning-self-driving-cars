"""RC Car Dataset."""

from torch.utils import data
from utils import augment


class Dataset(data.Dataset):

    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __getitem__(self, index):
        batch_samples = self.samples[index]
        steering_angle = float(batch_samples[3])

        center_img, steering_angle_center = augment(
            batch_samples[0], steering_angle)
        left_img, steering_angle_left = augment(
            batch_samples[1], steering_angle + 0.4)
        right_img, steering_angle_right = augment(
            batch_samples[2], steering_angle - 0.4)
        center_img = self.transform(center_img)
        left_img = self.transform(left_img)
        right_img = self.transform(right_img)

        return (center_img, steering_angle_center), (left_img, steering_angle_left), (right_img, steering_angle_right)

    def __len__(self):
        return len(self.samples)
