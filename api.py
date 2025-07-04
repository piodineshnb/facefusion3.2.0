import os
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import subprocess
from facefusion.execution import get_available_execution_providers

app = FastAPI()

TEMP_DIR = "temp"
os.makedirs(TEMP_DIR, exist_ok=True)

class FaceSwapRequest(BaseModel):
    mediaUrl: str
    faces: List[dict]

def download_file(url, dest_path):
    r = requests.get(url, stream=True)
    if r.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(1024):
                f.write(chunk)
    else:
        raise Exception(f"Failed to download {url}")

@app.post("/swap")
async def swap_faces(request: FaceSwapRequest):
    try:
        target_url = request.mediaUrl
        job_id = str(uuid.uuid4())
        # Download the original video only once
        original_target_path = os.path.join(TEMP_DIR, f"target_{job_id}.mp4")
        download_file(target_url, original_target_path)

        # Download all sources and references
        source_paths = []
        reference_paths = []
        for face in request.faces:
            source_url = face["source"]
            reference_url = face["target"]

            source_id = str(uuid.uuid4())
            source_path = os.path.join(TEMP_DIR, f"source_{source_id}.jpg")
            reference_path = os.path.join(TEMP_DIR, f"reference_{source_id}.jpg")

            download_file(source_url, source_path)
            download_file(reference_url, reference_path)

            source_paths.append(source_path)
            reference_paths.append(reference_path)

        # Sequentially process each (source, reference) pair
        input_video = original_target_path
        for i, (source_path, reference_path) in enumerate(zip(source_paths, reference_paths)):
            output_path = os.path.join(TEMP_DIR, f"result_{job_id}_{i}.mp4")
            cmd = [
                r"D:\facefusion\venv\Scripts\python.exe", "facefusion.py", "headless-run",
                "--target", input_video,
                "--output-path", output_path,
                "--face-enhancer-model", "gfpgan_1.4",
                "--face-enhancer-blend", "100",
                "--face-enhancer-weight", "1.0",
                "--output-video-encoder", "libx264",
                "--output-video-quality", "95",
                "--execution-providers", "cuda",
                "--source", source_path,
                "--reference-face-path", reference_path
            ]
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            process.wait()
            if process.returncode != 0:
                raise Exception(f"FaceFusion failed with code {process.returncode} on pair {i+1}")
            # Use the output as input for the next run
            input_video = output_path

        print("API available providers:", get_available_execution_providers())
        # Return the final output
        return {"output": input_video}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with python api.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
