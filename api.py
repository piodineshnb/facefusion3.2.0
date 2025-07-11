import os
import uuid
import requests
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List
import subprocess
from facefusion.execution import get_available_execution_providers
from utils.check_and_hold_user_credits import check_and_hold_user_credits
from utils.deduct_user_credits import deduct_user_credits
from utils.refund_user_credits import refund_user_credits
from utils.register_video_swap_document import register_video_swap_document
from utils.update_video_swap_document import update_video_swap_document
from utils.register_error import register_error
from utils.download_file_from_url import download_file_from_url
from utils.upload_file import upload_file
from utils.remove_file import remove_file
from utils.update_progress import ProgressUpdater, update_progress
from utils.update_swap_status_local import update_swap_status_local
from utils.custom_exception import CustomException
from utils.get_doc_data import get_doc_data
from utils.get_user_data import get_user_data
from utils.get_plan_data import get_plan_data
from utils.create_webm_from_video import create_webm_from_video
from utils.webp_handler import webp_to_mp4, mp4_to_webp
from utils.is_video_check import is_video_check
import threading
from fastapi.middleware.cors import CORSMiddleware
from cachetools import TTLCache
import uvicorn
import facefusion.globals as globals
from facefusion.globals import progress_cache
import time
import facefusion.logger as logger
from tqdm import tqdm

from facefusion.vision import detect_video_fps
import cv2

def get_total_frames(video_path, trim_frame_start=0, trim_frame_end=None):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if trim_frame_end is not None:
        total = min(total, trim_frame_end)
    return max(0, total - trim_frame_start)

# Later, when you know the video path:
video_path = "path/to/your/video.mp4"  # set this to your actual video path
total_frames = get_total_frames(video_path)
progress_bar = tqdm(total=total_frames, desc="FaceFusion", unit="frame", ascii=True)
progress_count = 0

def progress_callback(n=1):
    global progress_count
    progress_count += n
    progress_bar.update(n)
    percent = int(progress_count / total_frames * 100)
    logger.info(f"Progress: {percent}%", "facefusion")

job_lock = threading.Lock()

app = FastAPI()
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up a TTLCache with a max size of 100 items and a TTL of 300 seconds (5 minutes)
jobs = TTLCache(maxsize=10, ttl=172800)

@app.get('/')
def home():
    return "ok"

# Global toggle for face enhancement
enable_face_enhance = True

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

@app.post("/set_face_enhance")
async def set_face_enhance(enable: bool):
    global enable_face_enhance
    enable_face_enhance = enable
    return {"enable_face_enhance": enable_face_enhance}

def process_swap_job(params):
    credits_required = None
    try:
        target_url = params.get("mediaUrl")
        job_id = params.get("jobId") or str(uuid.uuid4())
        user_id = params.get("userId") or (params["faces"][0].get("user_id") if params.get("faces") and isinstance(params["faces"][0], dict) else 'unknown')
        project_id = params.get("projectId") or (params["faces"][0].get("project_id") if params.get("faces") and isinstance(params["faces"][0], dict) else 'default')
        swap_doc_id = params.get("swapId") or job_id
        duration_in_seconds = 0  # Set appropriately if you have this info

        steps_total = 3 + len(params["faces"])  # 1: download, N: swaps, 1: upload, 1: cleanup
        current_step = 0
        progress_percent = 0
        progress_bar = tqdm(total=100, desc="Progress", unit="%", ascii=True)
        def update_progress_bar(percent):
            nonlocal progress_percent
            progress_bar.update(percent - progress_percent)
            progress_percent = percent
            logger.info(f"Progress: {percent}%", __name__)

        print("[DEBUG] Registering swap document...")
        register_video_swap_document(user_id, swap_doc_id, params["faces"], project_id)
        print("[DEBUG] Swap document registered.")
        current_step += 1
        update_progress_bar(int(current_step / steps_total * 100))

        print("[DEBUG] Checking and holding user credits...")
        credits_required = check_and_hold_user_credits(user_id, swap_doc_id, duration_in_seconds, project_id)
        print(f"[DEBUG] Credits held: {credits_required}")

        TEMP_DIR = "temp"
        os.makedirs(TEMP_DIR, exist_ok=True)
        original_target_path = os.path.join(TEMP_DIR, f"target_{job_id}.mp4")
        print(f"[DEBUG] Downloading target video to {original_target_path}...")
        download_file_from_url(target_url, original_target_path)
        print("[DEBUG] Target video downloaded.")
        current_step += 1
        update_progress_bar(int(current_step / steps_total * 100))

        source_paths = []
        reference_paths = []
        for face in params["faces"]:
            source_url = face["source"]
            reference_url = face["target"]
            source_id = str(uuid.uuid4())
            source_path = os.path.join(TEMP_DIR, f"source_{source_id}.jpg")
            reference_path = os.path.join(TEMP_DIR, f"reference_{source_id}.jpg")
            print(f"[DEBUG] Downloading source: {source_url} to {source_path}")
            download_file_from_url(source_url, source_path)
            print(f"[DEBUG] Downloading reference: {reference_url} to {reference_path}")
            download_file_from_url(reference_url, reference_path)
            source_paths.append(source_path)
            reference_paths.append(reference_path)
        print("[DEBUG] All sources and references downloaded.")

        progress_updater = ProgressUpdater(update_interval=5, min_percentage_increment=5, user_id=user_id, session_id=swap_doc_id)
        progress_updater.set_total(len(source_paths))

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
                "--reference-face-path", reference_path,
                "--processors", "face_swapper", "face_enhancer",
                "--face-selector-mode", "reference"
            ]
            env = os.environ.copy()
            env["ENABLE_FACE_ENHANCE"] = "1"
            print(f"[DEBUG] Running facefusion.py for pair {i+1}...")
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                env=env
            )
            for line in iter(process.stdout.readline, ''):
                print(line.strip())
            process.wait()
            if process.returncode != 0:
                print(f"[ERROR] FaceFusion failed with code {process.returncode} on pair {i+1}")
                register_error(user_id, swap_doc_id, {"error": f"FaceFusion failed with code {process.returncode} on pair {i+1}"}, project_id)
                if credits_required is not None:
                    refund_user_credits(user_id, credits_required, project_id)
                update_swap_status_local('failed')
                progress_bar.close()
                return {"error": f"FaceFusion failed with code {process.returncode} on pair {i+1}"}, 500
            update_progress(i+1, progress_updater)
            input_video = output_path
            current_step += 1
            update_progress_bar(int(current_step / steps_total * 100))
        print("[DEBUG] Video processing complete.")

        # Try upload and update steps, catch and log exceptions, only refund if these fail
        try:
            print("[DEBUG] Uploading result video...")
            result_url = upload_file(input_video, f"result_{job_id}.mp4", ".mp4", project_id=project_id)
            print(f"[DEBUG] Result uploaded: {result_url}")

            print(f"[DEBUG] About to update swap document: user_id={user_id}, result_url={result_url}, swap_doc_id={swap_doc_id}")
            update_video_swap_document(user_id, result_url, swap_doc_id, preview_url=None, total_time=0, upload_time=0, thumbnail_url=None, project_id=project_id)
            print("[DEBUG] Swap document updated.")

            print("[DEBUG] Deducting credits...")
            if credits_required is not None:
                deduct_user_credits(user_id, credits_required, project_id)
            print("[DEBUG] Credits deducted.")

            update_swap_status_local('completed')
            current_step += 1
            update_progress_bar(int(current_step / steps_total * 100))
        except Exception as e:
            print(f"[ERROR] Exception after processing: {e}")
            register_error(user_id, swap_doc_id, {"error": str(e)}, project_id)
            if credits_required is not None:
                refund_user_credits(user_id, credits_required, project_id)
            update_swap_status_local('failed')
            progress_bar.close()
            return {"error": str(e)}, 500

        # Clean up temp files
        print("[DEBUG] Cleaning up temp files...")
        for path in [original_target_path, *source_paths, *reference_paths, input_video]:
            try:
                remove_file(path)
            except Exception as cleanup_e:
                print(f"[WARNING] Failed to remove file {path}: {cleanup_e}")
        print("[DEBUG] Cleanup complete.")
        current_step += 1
        update_progress_bar(100)
        progress_bar.close()

        return {"output": result_url}, 200

    except CustomException as ce:
        print(f"[ERROR] CustomException: {ce}")
        register_error(user_id, swap_doc_id, {"error": ce.message}, project_id)
        if credits_required is not None:
            refund_user_credits(user_id, credits_required, project_id)
        update_swap_status_local('failed')
        return {"error": ce.message}, 400
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        register_error(user_id, swap_doc_id, {"error": str(e)}, project_id)
        if credits_required is not None:
            refund_user_credits(user_id, credits_required, project_id)
        update_swap_status_local('failed')
        return {"error": str(e)}, 500

def execute_job(job_id, params):
    with job_lock:
        try:
            swap_doc_id = params.get("swapId")
            project_id = params.get('projectId', "default")
            print(f"Executing job {job_id} and swapId {swap_doc_id}")
            response, status_code = process_swap_job(params)
            if status_code != 200:
                jobs[job_id] = {"status": "FAILED", **response, "swapId": swap_doc_id, "projectId": project_id}
            else:
                jobs[job_id] = {"status": "COMPLETED", "output": response, "swapId": swap_doc_id, "projectId": project_id}
        except Exception as e:
            jobs[job_id] = {"status": "FAILED", "error": {"code": 500, "message": str(e)}}

@app.post("/swap")
def swap(params: dict, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    swap_doc_id = params.get("swapId")
    project_id = params.get('projectId', "default")
    if swap_doc_id:
        progress_cache[swap_doc_id] = 0
    jobs[job_id] = {
        "status": "IN_PROGRESS",
        "swapId": swap_doc_id,
        "projectId": project_id,
        "userId": params.get("userId"),
        "progress": 0
    }
    background_tasks.add_task(execute_job, job_id, params)
    return {"jobId": job_id}

def get_progress(swap_doc_id):
    try:
        if swap_doc_id in progress_cache:
            return {'progress': progress_cache[swap_doc_id]}
        else:
            return {'error': 'Session ID not found'}, 404
    except Exception as e:
        return {'error': str(e)}, 500

@app.get('/status')
def status():
    try:
        # If there are jobs, return the most recent or currently running job's status
        if len(jobs) > 0:
            last_job_id = list(jobs.keys())[-1]
            job = jobs[last_job_id]
            # Sync progress from progress_cache if available
            if "swapId" in job and job["swapId"] in progress_cache:
                job["progress"] = progress_cache[job["swapId"]]
            # Build the desired response format
            response = {
                "status": job.get("status", "processing").lower(),
                "swapId": job.get("swapId"),
                "userId": job.get("userId") or globals.user_id or (job.get("params", {}) or {}).get("userId"),
                "projectId": job.get("projectId", globals.project_id),
                "progress": job.get("progress", 0),
                "mediaUrl": None,
                "lastUpdated": job.get("lastUpdated") or getattr(globals, 'last_updated', None) or int(time.time() * 1000),
            }
            # Set mediaUrl if available in job output
            if isinstance(job.get("output"), dict) and "output" in job["output"]:
                response["mediaUrl"] = job["output"]["output"]
            elif isinstance(job.get("output"), str):
                response["mediaUrl"] = job["output"]
            # Set progress to 100 if completed
            if response["status"] == "completed":
                response["progress"] = 100
            # Set credits: use job['credits'] if present, else globals.credits, else 1 if completed
            if "credits" in job:
                response["credits"] = job["credits"]
            elif hasattr(globals, 'credits') and globals.credits:
                response["credits"] = globals.credits
            elif response["status"] == "completed":
                response["credits"] = 1
            return response
        # Fallback to globals if no jobs in cache
        progress = None
        if globals.session_id:
            progress_response = get_progress(globals.session_id)
            if 'progress' in progress_response:
                progress = progress_response['progress']
            result = {'status': globals.swap_status, 'swapId': globals.session_id, 'userId': globals.user_id,'projectId': globals.project_id,
                    'progress': progress, "mediaUrl": globals.output_media_url, "lastUpdated": globals.last_updated}
            if globals.swap_status == "completed" or progress == 100:
                result["credits"] = globals.credits
            return result
        return {'status': 'ok'}
    except Exception as e:
        return {'error': {'code': 500, 'message': str(e)}}, 500

@app.get("/status/{job_id}")
def get_job(job_id: str):
    if job_id in jobs:
        progress = get_progress(jobs[job_id]["swapId"])
        result = jobs[job_id]
        result["progress"] = progress
        return result
    else:
        return {'error': {'code': 500, 'message': 'Invalid or Expired Job Id'}}, 500

# Run with python api.py
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
