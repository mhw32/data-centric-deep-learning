import time
import torch
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms

from celery import Task, Celery

from src.system import DigitClassifierSystem
from consts import MODEL_PATH, BROKER_URL, BACKEND_URL

# This creates a Celery instance: we specify it to look at the 
# `tasks.py` file and to use Redis as the broker and backend.
app = Celery('tasks', broker=BROKER_URL, backend=BACKEND_URL)


class PredictionTask(Task):
  r"""Celery task to load a pretrained PyTorch Lightning system."""
  abstract = True

  def __init__(self):
    super().__init__()
    self.system = None

  def __call__(self, *args, **kwargs):
    r"""Load lightning system on first call. This way we do not need to 
    load the system on every task request, which quickly gets expensive.
    """
    if self.system is None:
      print(f'Loading digit classifier system from {MODEL_PATH}')
      self.system = self.get_system()
      print('Loading successful.')

    # pass arguments through 
    return self.run(*args, **kwargs)
  
  def get_system(self):
    system = None
    # ================================
    # FILL ME OUT
    # 
    # Load the checkpoint in `MODEL_PATH` using the class 
    # `DigitClassifierSystem`. Store in the variable `system`.
    # 
    # Pseudocode:
    # --
    # system = ...
    # 
    # Types:
    # --
    # system: DigitClassifierSystem
    # ================================
    assert system is not None, "System is not loaded."
    return system.eval()


@app.task(ignore_result=False,
          bind=True,
          base=PredictionTask)
def predict_single(self, data):
  r"""Defines what `PredictionTask.run` should do.
  
  In this case, it will use the loaded LightningSystem to compute
  the forward pass and make a prediction.

  Argument
  --------
  data (str): url denoting image path to do prediction for.

  Returns
  -------
  results (dict[str, any]): response dictionary.
    probs (list[float]): list of probabilities for each MNIST class.
    label (int): predicted class (one with highest probability).
  """
  # image I/O can be very expensive. Put this in worker so we don't 
  # stall on it in FastAPI
  im = Image.open(data)
  im: Image = im.convert('L')  # convert to grayscale
  im_transforms = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
  ])
  im = im_transforms(im)
  im = im.unsqueeze(0)

  # default (placeholder) values
  results = {'label': None, 'probs': None}

  with torch.no_grad():
    logits = None
    # ================================
    # FILL ME OUT
    # 
    # Copy over your solution from `week3_fastapi/api.py`.
    # 
    # Pseudocode:
    # --
    # logits = ... (use system)
    # 
    # Types:
    # --
    # logits: torch.Tensor (shape: 1x10)
    # ================================
    assert logits is not None, "logits is not defined."

    # To extract the label, just find the largest logit.
    label = torch.argmax(logits, dim=1)  # shape (1)
    label = label.item()                 # tensor -> integer

    probs = None
    # ================================
    # FILL ME OUT
    # 
    # Copy over your solution from `week3_fastapi/api.py`.
    # 
    # Pseudocode:
    # --
    # probs = ...do something to logits...
    # 
    # Types:
    # --
    # probs: torch.Tensor (shape: 1x10)
    # ================================
    assert probs is not None, "probs is not defined."
    probs = probs.squeeze(0)        # squeeze to (10) shape
    probs = probs.numpy().tolist()  # convert tensor to list

  results['probs'] = probs
  results['label'] = label

  # ================================
  # NOTE: simulate hard computation! This will help motivate 
  # why we need Celery.
  # 
  # Uncomment me when you are told to in the notes!
  # time.sleep(5)
  # ================================

  return results
