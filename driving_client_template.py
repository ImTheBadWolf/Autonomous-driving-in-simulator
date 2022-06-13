import drive

driver = drive.Driver()

def driving_callback(data):
  """Callback for telemetry data processing

  Activated when telemetry data is received from a socket.
  @param data Telemetry data.
  speed = data["speed"]
  throttle = data["throttle"]
  steering = data["steering_angle"]
  img = data["image"]
  """
  driver.send_control(0,0)  #steering, throttle

if __name__ == '__main__':
  driver.set_driving_callback(driving_callback)
  driver.run()