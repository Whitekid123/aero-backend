from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import numpy as np
import io
import gc

# Limit PyTorch threads to save RAM on Render Free Tier
torch.set_num_threads(1)

# 1. Define the PointNet Architecture
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

# CORS Middleware (Allows your Vercel frontend to talk to your Render backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return {"status": "Aerodynamic AI Backend is Live!", "model_loaded": True}

@app.post("/predict")
async def predict_drag(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        file_obj = io.BytesIO(contents)
        
        mesh = trimesh.load(file_obj, file_type='stl', force='mesh')
        
        if len(mesh.vertices) == 0:
            return JSONResponse(status_code=400, content={"error": "The uploaded STL file contains no vertices."})
            
        points = mesh.sample(1024)
        
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.linalg.norm(points, axis=1))
        if max_dist == 0: max_dist = 1
        points = points / max_dist
        
        point_cloud = torch.tensor(points, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
        
        with torch.no_grad():
            prediction = model(point_cloud).item()
            
        # FREE RAM: Delete heavy variables to prevent Render Free Tier crash
        del mesh, points, point_cloud
        gc.collect()
        
        if prediction > 0.35:
            feedback = "Engineering Insight: The geometry exhibits high aerodynamic resistance. It is recommended to evaluate the underbody topology and rear taper to reduce turbulent wake formation."
        else:
            feedback = "Engineering Insight: The geometry demonstrates optimal aerodynamic efficiency. The current macroscopic topology is suitable for preliminary design screening."
        
        return {
            "predicted_cd": prediction,
            "engineering_feedback": feedback
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# 4. Cloud Deployment Startup Block
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)