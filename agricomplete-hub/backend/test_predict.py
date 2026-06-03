import urllib.request
import os

boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
img_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'data for model training',
    'Plant_leave_diseases_dataset_without_augmentation',
    'Tomato___Late_blight',
    'image (1).JPG'
)

print(f"Testing with image: {img_path}")
print(f"File exists: {os.path.exists(img_path)}")

with open(img_path, 'rb') as f:
    img_data = f.read()

body = b''
body += ('--' + boundary + '\r\n').encode()
body += b'Content-Disposition: form-data; name="image"; filename="test.jpg"\r\n'
body += b'Content-Type: image/jpeg\r\n\r\n'
body += img_data
body += ('\r\n--' + boundary + '--\r\n').encode()

req = urllib.request.Request(
    'http://localhost:5000/api/disease/predict',
    data=body,
    headers={'Content-Type': 'multipart/form-data; boundary=' + boundary},
    method='POST'
)

try:
    resp = urllib.request.urlopen(req)
    print(f"Status: {resp.status}")
    print(f"Response: {resp.read().decode()}")
except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code}")
    print(f"Response: {e.read().decode()}")
except Exception as e:
    print(f"Error: {e}")
