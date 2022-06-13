import drive
import keyboard
import base64
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from PIL import Image
from flask import Flask
from io import BytesIO
from keras.models import load_model
import utils
import cv2

config = utils.load_config("config.json")
driver = drive.Driver()

#Maximal allowed speed of the car
MAX_SPEED = config["maxSpeed"]
COMPENSATOR = MAX_SPEED * -0.17

keyboard.on_press_key("+", lambda _: change_max_speed("+"))
keyboard.on_press_key("-", lambda _: change_max_speed("-"))

def change_max_speed(sign):
    """Change maximal speed limit while driving

    @param sign "+" or "-" to increase or decrease the speed limit.
    """
    global MAX_SPEED
    global COMPENSATOR
    MAX_SPEED += 0.2 if sign == "+" else -0.2
    COMPENSATOR = MAX_SPEED * -0.17


def driving_callback(data):
  """Callback for telemetry data processing

  Activated when telemetry data is received from a socket.
  @param data Telemetry data.
  speed = data["speed"]
  throttle = data["throttle"]
  steering = data["steering_angle"]
  img = data["image"]
  """
  if data:
    speed = float(data["speed"])
    throttle = float(data["throttle"])
    steering = float(data["steering_angle"])
    img = Image.open(BytesIO(base64.b64decode(data["image"])))

    try:
        img = np.asarray(img)
        preprocessed_image = utils.preprocess(img)
        img = np.array([preprocessed_image])

        if config["showCameraPreview"]:
            cv2.imshow('Upscaled preview', cv2.resize(preprocessed_image, (320, 160), cv2.INTER_AREA))
            cv2.waitKey(1)

        steering = float(model.predict(img, batch_size=1))
        throttle = ((MAX_SPEED-COMPENSATOR-speed) / 10)*.5  if (MAX_SPEED-COMPENSATOR-speed) > 0 else 0

        #Prevent overspeeding at start
        if throttle > 0.2:
            throttle = 0.2
        if config["verboseTelemetry"]:
            print('Steering: {:.4f} | Throttle: {:.4f}'.format(steering, throttle))
        driver.send_control(steering, throttle)
    except Exception as e:
        print(e)
    else:
        driver.manual()

if __name__ == '__main__':
  model = load_model(config["drivingModel"])
  driver.set_driving_callback(driving_callback)
  driver.run()