"""
Building a REST API servicing a neural network checkpoint using FastAPI.
"""
import os
from os.path import join
from typing import List, Optional, Dict, Any

import urllib.request
from pathlib import Path
from fastapi import FastAPI
from fastapi import Request
from pydantic import BaseModel, Field

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from src.system import DigitClassifierSystem

# this is a special path for the deployed model
MODEL_PATH: str = join(Path(__file__).parent, "ckpts/deploy.ckpt")


class InferenceInput(BaseModel):
  r"""Input values for model inference. We will expect the image to be passed
  to us as a URL that we download.
  """
  image_url: str = Field(..., 
    example = 'https://machinelearningmastery.com/wp-content/uploads/2019/02/sample_image.png', 
    title = 'url to handwritten digit')


class InferenceResult(BaseModel):
  r"""Inference outputs from the model."""
  # Two expected return fields: a label (integer) and a probability 
  # vector - a list of 10 numbers that sum to one.
  label: int = Field(..., example = 0, title = 'Predicted label for image')
  probs: List[float] = Field(..., example = [0.1] * 10, 
    title='Predicted probability for predicted label')


class InferenceResponse(BaseModel):
  r"""Output response for model inference."""
  error: str = Field(..., example=False, title='error?')
  results: Dict[str, Any] = Field(..., example={}, 
    title='label and probability results')


class ErrorResponse(BaseModel):
  r"""Error response for the API."""
  error: str = Field(..., example=True, title='error?')
  message: str = Field(..., example='', title='error message')
  traceback: Optional[str] = Field(None, example='', title='detailed traceback of the error')


app: FastAPI = FastAPI(
  title = 'mnist classifier',
  description = 'corise data-centric deep learning week 3',
)


@app.on_event("startup")
async def startup_event():
  r"""Initialize FastAPI."""
  print(f'Loading digit classifier system from {MODEL_PATH}')
  system = DigitClassifierSystem.load_from_checkpoint(MODEL_PATH)
  system.eval()
  print('Loading successful.')

  app.package = {'system': system}


@app.post('/api/v1/predict',
  response_model = InferenceResponse,
  responses = {
    422: {'model': ErrorResponse},
    500: {'model': ErrorResponse}})
def predict(request: Request, body: InferenceInput):
  print('`/api/v1/predict` endpoint called.')

  # download image to cache
  image_name: str = os.path.basename(body.image_url)
  cache_dir: str = os.path.abspath('artifacts/cache')
  os.makedirs(cache_dir, exist_ok = True)
  local_path: str  = os.path.abspath(join(cache_dir, image_name))

  # add hacks to download images as a bot
  opener = urllib.request.build_opener()
  opener.addheaders = [('User-Agent',
    'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/36.0.1941.0 Safari/537.36')]
  urllib.request.install_opener(opener)
  # download locally: we assume that the image is already preprocessed.
  urllib.request.urlretrieve(body.image_url, local_path)

  system = app.package['system']
  results: Dict[str, Any] = {'label': None, 'probs': None}
  
  im: Image = Image.open(local_path)
  im: Image = im.convert('L')  # convert to grayscale

  # make sure the image is the right size
  im_transforms = transforms.Compose([
    transforms.Resize(28),
    transforms.CenterCrop(28),
    transforms.ToTensor(),
  ])
  
  # HINT: Don't forget that the `system` expects minibatches. Even if 
  # you are only passing in one element, it must be of shape 1x1x28x28.
  im = im_transforms(im)
  im = im.unsqueeze(0)

  with torch.no_grad():
    logits = None

    # ================================
    # FILL ME OUT
    # 
    # Use `system` to make a prediction on the input `im` and
    # save it to the variable `logits`. The output should be of
    # shape `(1, 10)`.
    # 
    # HINT: there is no data module here. Which method should you use
    # from system to make a prediction? 
    # 
    # Our solution is one of code. 
    # 
    # Pseudocode:
    # --
    # logits = ... (use system)
    # 
    # Types:
    # --
    # logits: torch.Tensor (shape: 1x10)
    # ================================

    # To extract the label, just find the largest logit.
    label = torch.argmax(logits, dim=1)  # shape (1)
    label = label.item()                 # tensor -> integer

    probs = None
    # ================================
    # FILL ME OUT
    # 
    # Normalize `logits` to probabilities and save it to the 
    # variable `probs`. Remember `logits` is shape (1, 10), and 
    # we expect your output `probs` to be shape (1, 10) as well.
    # 
    # Pseudocode:
    # --
    # probs = ...do something to logits...
    # 
    # Types:
    # --
    # probs: torch.Tensor (shape: 1x10)
    # ================================
    probs = probs.squeeze(0)        # squeeze to (10) shape
    probs = probs.numpy().tolist()  # convert tensor to list

  results['label'] = label
  results['probs'] = probs

  os.remove(local_path)  # delete cached file

  return {
    'error': False,
    'results': results,
  }
