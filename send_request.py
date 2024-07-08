import requests

# The URL where your Flask app is running
url = 'http://localhost:5000/predict'

# The path to the image you want to send
image_path = 'C:/Users/Sara/Desktop/FashionExpert-SE/FashionExpert/image9.jpeg'

# Open the image in binary mode and send it in a POST request
with open(image_path, 'rb') as f:
    files = {'file': f}
    response = requests.post(url, files=files)

# Print out the response from the server
print(response.json())
