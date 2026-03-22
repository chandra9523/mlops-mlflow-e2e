import requests

def test_api():
    res = requests.get("http://127.0.0.1:8000/")
    assert res.status_code == 200