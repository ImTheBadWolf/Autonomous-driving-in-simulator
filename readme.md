# Autonomous driving in simulator

Use python version = 3.6.5


Updating pip may be needed
## Create and activate venv
- python -m venv .venv
- Set-ExecutionPolicy Unrestricted -Scope Process (windows only)
- .\.venv\Scripts\activate

## Install dependencies

- pip install -r requirements.txt

## Training

- Generate dataset using recording function in simulator
- Change `learningDataDir` in [config.json](config.json)
- Run `python train.py`

## Autonomous driving

- Launch [beta_simulator.exe](simulator/beta_simulator.exe)
- Select Autonomous mode
- Change `drivingModel` in [config.json](config.json) to the trained model to be used
- Run `python autonomous_driving_client.py`
