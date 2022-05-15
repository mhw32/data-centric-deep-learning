"""
Building a REST API servicing a neural network checkpoint using FastAPI.
"""
import os
from os.path import join

import urllib.request
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from celery.result import AsyncResult
from tasks import predict_single
from src.schema import InferenceInput, TaskResponse, InferenceResponse, \
  ErrorResponse


app: FastAPI = FastAPI(
  title = 'mnist classifier',
  description = 'corise data-centric deep learning week 3',
)


@app.post('/api/v1/predict',
  response_model=TaskResponse,
  responses={
    422: {'model': ErrorResponse},
    500: {'model': ErrorResponse}})
async def predict(request: Request, body: InferenceInput):
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

  task_id = predict_single.delay(local_path)
  return {
    'task_id': str(task_id),
    'status': 'processing',
  }


@app.get('/api/v1/result/{task_id}',
  response_model=InferenceResponse,
  responses={
    422: {'model': ErrorResponse},
    500: {'model': ErrorResponse}})
async def get_result(task_id):
  r"""Fetch result for given `task_id`."""
  print(f'`/api/v1/result/{task_id}` endpoint called.')

  task = AsyncResult(task_id)

  if not task.ready():
    return JSONResponse(
      status_code=202, 
      content={
        'task_id': task_id, 
        'status': 'processing'
      })
  
  results = task.get()
  
  return {
    'task_id': task_id, 
    'status': 'complete', 
    'results': results,
  }
