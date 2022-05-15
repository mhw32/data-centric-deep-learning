from os.path import join
from pathlib import Path

MODEL_PATH: str = join(Path(__file__).parent, "ckpts/deploy.ckpt")

# Use redis as the broker
BROKER_URL: str = 'redis://localhost:6379/0'

# Use redis as the backend cache as well
BACKEND_URL: str = 'redis://localhost:6379/0'
