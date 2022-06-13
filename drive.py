import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
class Driver:
    sio = socketio.Server()
    driving_callback = {}
    def __init__(self):
        pass

    def set_driving_callback(self, driving_callback):
        self.driving_callback = driving_callback

    def run(self):
        self.call_backs()
        self.app = socketio.Middleware(self.sio, Flask(__name__))
        eventlet.wsgi.server(eventlet.listen(('', 4567)), self.app)

    def call_backs(self):
        @self.sio.on('connect')
        def connect(sid, environ):
            """Connection event handler."""
            print("Connected ", sid)
            self.send_control(0, 0)

        @self.sio.on('telemetry')
        def telemetry(sid, data):
            self.driving_callback(data)

    def manual(self):
        self.sio.emit('manual', data={}, skip_sid=True)

    def send_control(self, steering_angle, throttle):
        """Send control signal to simulator.

        Sends control signal to simulator upon which the car moves and turns.
        @param steering_angle Steering angle of wheels.
        @param throttle Throttle to apply.
        """
        self.sio.emit(
            "steer",
            data={
                'steering_angle': steering_angle.__str__(),
                'throttle': throttle.__str__()
            },
            skip_sid=True)