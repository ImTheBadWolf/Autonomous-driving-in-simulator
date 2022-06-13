import cv2, os
import json
import numpy as np
import matplotlib.image as mpimg


def load_config(config_file):
    """Load json configuration file.

    @param config_file Path to json configuration file.
    """
    with open(config_file, "r") as jsonfile:
        return json.load(jsonfile)

config = load_config("config.json")

def load_image(data_dir, image_file):
    """Load image from file directory."""

    image = mpimg.imread(os.path.join(data_dir, image_file.strip()))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def crop(image):
    """Crop the image to remove sky and front of the car."""
    return image[60:-25, :, :]


def resize(image):
    """Resize the image to match shape required by network model"""
    return cv2.resize(image, (config["imageWidth"], config["imageHeight"]), cv2.INTER_AREA)


def rgb2yuv(image):
    """Convert the image from RGB to YUV."""
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """Preprocess the image.

    Apply crop, resize and convert from RGB to YUV.
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image


def choose_image(data_dir, center, left, right, steering_angle):
    """Randomly choose an image from one of the cameras, and adjust the steering angle accordingly."""

    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle


def random_flip(image, steering_angle):
    """Randomly flip the image and adjust the steering angle accordingly."""

    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    """Randomly shift the image vertically and horizontally."""
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_shadow(image):
    """Adds random shadow to image"""

    x1, y1 = config["imageWidth"] * np.random.rand(), 0
    x2, y2 = config["imageWidth"] * np.random.rand(), config["imageHeight"]
    xm, ym = np.mgrid[0:config["imageHeight"]*2, 0:config["imageWidth"]*2]
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """Randomly adjust brightness of the image."""

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def augument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    """Generate an augumented image and adjust steering angle."""

    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
    image, steering_angle = random_flip(image, steering_angle)
    image, steering_angle = random_translate(image, steering_angle, range_x, range_y)
    image = random_shadow(image)
    image = random_brightness(image)
    return image, steering_angle


def generate_batch(data_dir, image_paths, steering_angles, batch_size, is_training):
    """Generate training images and steering angles"""

    images = np.empty([batch_size, config["imageHeight"], config["imageWidth"], 3])
    steers = np.empty([batch_size, 2])
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            center, left, right = image_paths[index]
            steering_angle = steering_angles[index]

            if is_training and np.random.rand() < 0.6:
                image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            else:
                image = load_image(data_dir, center)

            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
