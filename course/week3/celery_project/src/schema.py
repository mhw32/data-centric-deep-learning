from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


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
  task_id: str = Field(..., example='1', title='task id')
  status: str = Field(..., example='processing')
  results: Dict[str, Any] = Field(..., example={}, 
    title='label and probability results')


class TaskResponse(BaseModel):
  task_id: str = Field(..., example='1', title='task id')
  status: str = Field(..., example='processing')


class ErrorResponse(BaseModel):
  r"""Error response for the API."""
  error: str = Field(..., example=True, title='error?')
  message: str = Field(..., example='', title='error message')
  traceback: Optional[str] = Field(None, example='', title='detailed traceback of the error')
