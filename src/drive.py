"""Test model."""

import argparse
import base64
from datetime import datetime
import os
import shutil

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO


import torch
from torch.autograd import Variable
import torchvision.transforms as transforms

from model import *

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

transformations = transforms.Compose(
    [transforms.Lambda(lambda x: (x / 127.5) - 1.0)])


class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 10
controller.set_desired(set_speed)


# MAX_SPEED = 15
# MIN_SPEED = 10
# speed_limit = MAX_SPEED


@sio.on('telemetry')
def telemetry(sid, data):

    if data:

        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])

        # The current throttle of the car
        throttle = float(data["throttle"])

        # The current speed of the car
        speed = float(data["speed"])

        image = Image.open(BytesIO(base64.b64decode(data["image"])))

        image_array = np.array(image.copy())
        image_array = image_array[65:-25, :, :]

        # transform RGB to BGR for cv2
        image_array = image_array[:, :, ::-1]
        image_array = transformations(image_array)
        image_tensor = torch.Tensor(image_array)
        image_tensor = image_tensor.view(1, 3, 70, 320)
        image_tensor = Variable(image_tensor)

        steering_angle = model(image_tensor).view(-1).data.numpy()[0]

        # throttle = controller.update(float(speed))

        # ----------------------- Improved by Siraj ----------------------- #
        # global speed_limit
        # if speed > speed_limit:
        #     speed_limit = MIN_SPEED
        # else:
        #     speed_limit = MAX_SPEED

        throttle = 1.2 - steering_angle ** 2 - (speed / set_speed) ** 2
        # ----------------------- Improved by Siraj ----------------------- #

        send_control(steering_angle, throttle)
        print("Steering angle: {} | Throttle: {}".format(
            steering_angle, throttle))

        # save frame
        if args.image_folder != '':
            timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
            image_filename = os.path.join(args.image_folder, timestamp)
            image.save('{}.jpg'.format(image_filename))
    else:
        # NOTE: DON'T EDIT THIS.
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)


if __name__ == '__main__':
    """Testing phase."""
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    parser.add_argument(
        'image_folder',
        type=str,
        nargs='?',
        default='',
        help='Path to image folder. This is where the images from the run will be saved.'
    )
    args = parser.parse_args()

    # define model
    # model = LeNet()
    model = NetworkNvidia()

    # check that model version is same as local PyTorch version
    try:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    except KeyError:
        checkpoint = torch.load(
            args.model, map_location=lambda storage, loc: storage)
        model = checkpoint['model']

    except RuntimeError:
        print("==> Please check using the same model as the checkpoint")
        import sys
        sys.exit()

    if args.image_folder != '':
        print("Creating image folder at {}".format(args.image_folder))
        if not os.path.exists(args.image_folder):
            os.makedirs(args.image_folder)
        else:
            shutil.rmtree(args.image_folder)
            os.makedirs(args.image_folder)
        print("RECORDING THIS RUN ...")
    else:
        print("NOT RECORDING THIS RUN ...")

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
