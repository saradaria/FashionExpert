from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.join(app.root_path, '../public')

num_classes = 6
# Example mapping, adjust based on your actual classes
idx_to_class = {0: "Belts", 1:"Dress", 2: "Jackets", 3: "Socks", 4:"Trousers", 5:"T-Shirt"}


model = models.mobilenet_v3_large(pretrained=False) 
model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, num_classes)

model.load_state_dict(torch.load('C:/Work/proiect-se/FashionExpert/trained_model.pth'))
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        img_transformed = transform(img)
        img_transformed = img_transformed.unsqueeze(0) 
        
        with torch.no_grad():
            output = model(img_transformed)
            _, predicted = torch.max(output, 1)
            predicted_index = predicted.item()
            predicted_class_name = idx_to_class[predicted_index]  # Map index to class name

        class_dir = os.path.join(BASE_DIR, predicted_class_name)
        os.makedirs(class_dir, exist_ok=True)  # Create the directory if it doesn't exist

        # Save the file in the corresponding class directory
        filename = secure_filename(file.filename)
        filepath = os.path.join(class_dir, filename)
        with open(filepath, 'wb') as f:
            f.write(img_bytes)

        print(predicted_class_name)
        return jsonify({'predicted_class': predicted_class_name})

@app.route('/files/<category_name>', methods=['GET'])
def list_files(category_name):
    category_path = os.path.join(BASE_DIR, category_name)
    if not os.path.exists(category_path):
        return jsonify({'error': f'Category {category_name} does not exist'}), 404

    # List only filenames without paths
    filenames = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]

    return jsonify(filenames)


if __name__ == '__main__':
    app.run(debug=True)
