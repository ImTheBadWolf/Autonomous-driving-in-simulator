import drive

driver = drive.Driver()
throttle = 0
incrementor = 0.02
flip = False

def driving_callback(data):
  """Callback for telemetry data processing

  Activated when telemetry data is received from a socket.
  @param data Telemetry data.
  speed = data["speed"]
  throttle = data["throttle"]
  steering = data["steering_angle"]
  img = data["image"]
  """
  global throttle
  global incrementor
  global flip

  if float(data["speed"]) > 7 and flip:
    incrementor *= -1
    flip = False

  if float(data["speed"]) < 6 and not flip:
    flip = True
  throttle += incrementor
  driver.send_control(0, throttle)

if __name__ == '__main__':
  driver.set_driving_callback(driving_callback)
  driver.run()