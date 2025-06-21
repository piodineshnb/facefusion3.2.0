import os
import uuid
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import subprocess

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
        # Use the first face in the list
        face = request.faces[0]
        source_url = face["source"]
        reference_url = face["target"]  # Reference face path from 'target'
        target_url = request.mediaUrl

        # Generate unique file names
        job_id = str(uuid.uuid4())
        source_path = os.path.join(TEMP_DIR, f"source_{job_id}.jpg")
        reference_path = os.path.join(TEMP_DIR, f"reference_{job_id}.jpg")
        target_path = os.path.join(TEMP_DIR, f"target_{job_id}.mp4")
        output_path = os.path.join(TEMP_DIR, f"result_{job_id}.mp4")

        # Download files
        download_file(source_url, source_path)
        download_file(reference_url, reference_path)
        download_file(target_url, target_path)

        # Build the FaceFusion command
        cmd = [
            "python", "facefusion.py", "headless-run",
            "--source", source_path,
            "--target", target_path,
            "--output-path", output_path,
            "--reference-face-path", reference_path,
            "--face-enhancer-model", "gfpgan_1.4",
            "--face-enhancer-blend", "100",
            "--face-enhancer-weight", "1.0",
            "--output-video-encoder", "libx264",
            "--output-video-quality", "95",
            "--execution-providers", "cuda"
        ]

        # MODIFIED: Run with real-time output printing
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stdout and stderr
            text=True,  # Return output as strings
            bufsize=1,  # Line buffered
            universal_newlines=True
        )
        
        # Print output in real-time
        for line in iter(process.stdout.readline, ''):
            print(line.strip())  # Print to console immediately
        
        # Wait for process to finish
        process.wait()
        
        if process.returncode != 0:
            raise Exception(f"FaceFusion failed with code {process.returncode}")

        return {"output": output_path}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run with python api.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)
