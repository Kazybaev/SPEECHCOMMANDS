import torchaudio
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import io
import soundfile as sf


labels = (torch.load("labels.pth"))
num_classes = len(labels)

class CheckAudio(nn.Module):
  def __init__(self):
    super().__init__()
    self.first = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((8, 8))
    )
    self.second = nn.Sequential(
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 128),
        nn.ReLU(),
        nn.Linear(128, num_classes)
    )

  def forward(self, x):
    x = x.unsqueeze(1)
    x = self.first(x)
    x = self.second(x)
    return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CheckAudio()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.MelSpectrogram(
    sample_rate = 16000,
    n_mels=32

)

# uvicorn main:app --reload
app = FastAPI()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        audio = torch.tensor(audio).unsqueeze(0)

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            audio = resampler(audio)

        spec = transform(audio).to(device)

        with torch.no_grad():
            outputs = model(spec)
            predicted = torch.argmax(outputs, dim=1).item()
            label = labels[predicted]

        return {"class": label, "index": predicted}

    except Exception as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
