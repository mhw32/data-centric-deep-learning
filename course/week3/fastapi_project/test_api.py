"""
Use PyTest to query the server. 
"""
import pytest
from fastapi.testclient import TestClient

from api import app

TEST_URL = 'https://conx.readthedocs.io/en/latest/_images/MNIST_44_0.png'


@pytest.fixture
def client():
  with TestClient(app) as c:
    yield c


def test_predict(client):
  """Test prediction response.

  Arguments
  ---------
  client (TestClient): see above.
  """
  headers = {}
  body = {'image_url': TEST_URL}
  response = client.post('/api/v1/predict',
    headers = headers, json=body)

  try:
    assert response.status_code == 200
    response_json = response.json()
    assert isinstance(response_json['results']['label'], int), \
      "`label` object has unexpected type."
    assert isinstance(response_json['results']['probs'], list), \
      "`probs` object has unexpected type."
    print('success!')

  except AssertionError:
    print(response.status_code)
    print(response.json())
    raise
