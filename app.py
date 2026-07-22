from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import io

# 1. Define the PointNet Architecture (Must match the training code)
class PointNetRegressor(nn.Module):
    def __init__(self):
        super(PointNetRegressor, self).__init__()
        self.mlp1 = nn.Conv1d(3, 64, 1)
        self.mlp2 = nn.Conv1d(64, 128, 1)
        self.mlp3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
    def forward(self, x):
        x = F.relu(self.bn1(self.mlp1(x)))
        x = F.relu(self.bn2(self.mlp2(x)))
        x = F.relu(self.bn3(self.mlp3(x)))
        x = torch.max(x, 2)[0]
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)
        return x

# 2. Load the trained model
model = PointNetRegressor()
model.load_state_dict(torch.load('best_pointnet_model.pth', map_location=torch.device('cpu')))
model.eval()

# 3. FastAPI Setup
app = FastAPI()

@app.get("/")
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Aerodynamic Drag Predictor</title>
        <style>
            body { font-family: Arial, sans-serif; text-align: center; margin-top: 50px; background-color: #f4f7f6; }
            .card { border: 1px solid #ccc; padding: 30px; display: inline-block; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); background: white; }
            button { background-color: #007BFF; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }
            button:hover { background-color: #0056b3; }
            h2 { color: #333; }
            .feedback { margin-top: 15px; color: #555; font-size: 14px; max-width: 400px; margin-left: auto; margin-right: auto; border-top: 1px solid #eee; padding-top: 15px; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>Artificial Intelligence Drag Prediction</h2>
            <p>Upload a 3D Vehicle Stereolithography File</p>
            <input type="file" id="fileInput" accept=".stl">
            <br><br>
            <button onclick="uploadFile()">Predict Drag Coefficient</button>
            <br><br>
            <h3 id="result" style="color: green;"></h3>
            <div id="feedback" class="feedback"></div>
        </div>

        <script>
            async function uploadFile() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                const feedbackDiv = document.getElementById('feedback');
                
                if (!fileInput.files[0]) {
                    alert("Please select an STL file first.");
                    return;
                }

                resultDiv.innerText = "Processing geometry and predicting...";
                feedbackDiv.innerText = "";

                const formData = new FormData();
                formData.append("file", fileInput.files[0]);

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    resultDiv.innerText = "Predicted Cd: " + data.predicted_cd.toFixed(4);
                    feedbackDiv.innerText = data.engineering_feedback;
                } catch (error) {
                    resultDiv.innerText = "Error processing file.";
                    console.error(error);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/predict")
async def predict_drag(file: UploadFile = File(...)):
    contents = await file.read()
    file_obj = io.BytesIO(contents)
    
    mesh = trimesh.load(file_obj, file_type='stl', force='mesh')
    points = mesh.sample(1024)
    
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.linalg.norm(points, axis=1))
    if max_dist == 0: max_dist = 1
    points = points / max_dist
    
    point_cloud = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
    
    with torch.no_grad():
        prediction = model(point_cloud).item()
        
    if prediction > 0.35:
        feedback = "Engineering Insight: The geometry exhibits high aerodynamic resistance. It is recommended to evaluate the underbody topology and rear taper to reduce turbulent wake formation."
    else:
        feedback = "Engineering Insight: The geometry demonstrates optimal aerodynamic efficiency. The current macroscopic topology is suitable for preliminary design screening."
    
    return {
        "predicted_cd": prediction,
        "engineering_feedback": feedback
    }

# 4. Cloud Deployment Startup Block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)