import os
import sys
import shutil
import base64
import numpy as np
import cv2
from typing import Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Ensure project root is on PYTHONPATH so we can import src modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.data.dataset_validator import DatasetValidator  # noqa: E402
from backend.services.inference_service import FullPipelineInferenceService  # noqa: E402

app = FastAPI(title="M2-ReID Backend", version="0.2.0")

# Allow Next.js dev server (port 3000) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure results directory exists and mount static files
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
app.mount("/results", StaticFiles(directory=RESULTS_DIR), name="results")

# ----------------------------------------------------------------------
# 1️⃣ Lazy-loaded Inference Service singleton
# ----------------------------------------------------------------------
_inference_service: Optional[FullPipelineInferenceService] = None

def get_inference_service() -> FullPipelineInferenceService:
    global _inference_service
    if _inference_service is None:
        _inference_service = FullPipelineInferenceService(device="cpu", conf_thresh=0.25)
    return _inference_service

# ----------------------------------------------------------------------
# 2️⃣ Health check
# ----------------------------------------------------------------------
@app.get("/api/health", response_model=dict)
def health_check():
    return {
        "status": "ok",
        "backend": "fastapi",
        "python": sys.version.split()[0],
        "models_ready": _inference_service is not None
    }

# ----------------------------------------------------------------------
# 3️⃣ Dataset validation (cached on startup)
# ----------------------------------------------------------------------
class DatasetStats(BaseModel):
    dataset: str
    root_path: str
    valid: bool
    train_images: int
    query_images: int
    gallery_images: int
    train_ids: int
    query_ids: int
    gallery_ids: int
    total_identities: int
    cameras: list[int]
    corrupted_images: int
    errors: list[str] | None = None

def _load_stats() -> dict:
    validator = DatasetValidator()
    return {"Market-1501": validator.validate_market1501(), "MSMT17": validator.validate_msmt17()}

VALIDATION_CACHE: dict = {}

@app.on_event("startup")
def preload():
    global VALIDATION_CACHE
    VALIDATION_CACHE = _load_stats()
    # Preload inference service in background
    try:
        get_inference_service()
    except Exception as e:
        print(f"[Startup Warning] Could not preload inference service: {e}")

@app.get("/api/datasets", response_model=list[str])
def list_datasets():
    return list(VALIDATION_CACHE.keys())

@app.get("/api/datasets/{name}", response_model=DatasetStats)
def get_dataset(name: str):
    stats = VALIDATION_CACHE.get(name)
    if not stats:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return DatasetStats(**stats)

@app.get("/api/datasets/{name}/cameras", response_model=dict)
def get_cameras(name: str):
    stats = VALIDATION_CACHE.get(name)
    if not stats:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"dataset": name, "camera_ids": stats["cameras"]}

# Camera topology endpoint
@app.get("/api/datasets/{name}/camera-topology", response_model=dict)
def get_topology(name: str):
    stats = VALIDATION_CACHE.get(name)
    if not stats:
        raise HTTPException(status_code=404, detail="Dataset not found")
    try:
        from src.temporal.camera_topology import CameraTopology  # noqa: E402
        cameras = stats.get("cameras", [])
        num_cams = len(cameras)
        topology = CameraTopology(num_cameras=num_cams)
        nodes = [str(c) for c in cameras]
        edges = []
        for i, c_src in enumerate(cameras):
            for c_dst in cameras:
                if c_src != c_dst:
                    prob = topology.get_transition_prior(c_src, c_dst)
                    if prob > 0.05:
                        pair = (c_src, c_dst)
                        mu, sigma = topology.travel_time_params.get(pair, (abs(c_src - c_dst) * 40.0, 30.0))
                        edges.append({
                            "source": str(c_src),
                            "target": str(c_dst),
                            "probability": round(prob, 4),
                            "mean_transit_s": round(mu, 1),
                            "std_transit_s": round(sigma, 1),
                        })
        return {"dataset": name, "nodes": nodes, "edges": edges}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# ----------------------------------------------------------------------
# 4️⃣ Video & Webcam Inference Endpoints
# ----------------------------------------------------------------------
class VideoInferenceRequest(BaseModel):
    video_path: str
    max_frames: Optional[int] = 200
    camera_id: Optional[int] = 1

class LiveFrameRequest(BaseModel):
    image_base64: str
    camera_id: Optional[int] = 0

@app.get("/api/inference/samples")
def get_sample_videos():
    """List available sample videos and pre-computed results."""
    user_downloads = os.path.expanduser("~/Downloads")
    samples = []
    
    # Check Downloads folder
    if os.path.exists(user_downloads):
        for f in os.listdir(user_downloads):
            if f.lower().endswith(".mp4"):
                full_p = os.path.join(user_downloads, f)
                samples.append({
                    "name": f,
                    "path": full_p.replace("\\", "/"),
                    "size_mb": round(os.path.getsize(full_p) / (1024 * 1024), 2)
                })
                
    # Check results folder
    cached_runs = []
    for f in os.listdir(RESULTS_DIR):
        if f.endswith(".json"):
            cached_runs.append(f)

    return {
        "sample_videos": samples,
        "results_available": cached_runs
    }

@app.get("/api/inference/precomputed-demo")
def get_precomputed_demo():
    """Returns the verified end-to-end run results on phani.mp4 instantly."""
    import json
    json_path = os.path.join(RESULTS_DIR, "phani_results.json")
    video_path = os.path.join(RESULTS_DIR, "phani_annotated.mp4")
    
    if not os.path.exists(json_path):
        raise HTTPException(status_code=404, detail="Precomputed demo results not found")

    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract keyframes from annotated video
    keyframes = []
    if os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_indices = [1, 25, 50, 75, 100, 130, 160, 190]
        cur_idx = 0
        while cap.isOpened() and cur_idx <= max(sample_indices):
            ret, frame = cap.read()
            if not ret:
                break
            cur_idx += 1
            if cur_idx in sample_indices:
                _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                b64 = base64.b64encode(buf).decode("utf-8")
                keyframes.append({
                    "frame_idx": cur_idx,
                    "timestamp": round(cur_idx / fps, 2),
                    "snapshot_b64": b64
                })
        cap.release()

    data["keyframes"] = keyframes
    data["video_url"] = "http://127.0.0.1:8001/results/phani_annotated.mp4"
    data["total_video_frames"] = 195
    data["fps"] = 24.0
    data["resolution"] = [1280, 720]
    return data

@app.post("/api/inference/run-video")
def run_video_inference(req: VideoInferenceRequest):
    """Run full YOLOv8 detection + Multi-Model Re-ID pipeline on a video path."""
    if not os.path.exists(req.video_path):
        raise HTTPException(status_code=400, detail=f"Video file not found at: {req.video_path}")
    
    service = get_inference_service()
    base_name = os.path.splitext(os.path.basename(req.video_path))[0]
    out_video_name = f"{base_name}_annotated.mp4"
    out_video_path = os.path.join(RESULTS_DIR, out_video_name)

    try:
        cid = req.camera_id or 1
        result = service.process_video(
            video_path=req.video_path,
            output_video_path=out_video_path,
            max_frames=req.max_frames or 200,
            camera_id=cid
        )
        result["video_url"] = f"http://127.0.0.1:8001/results/{out_video_name}"

        # Register detected persons into Forensic Search pool & gallery
        try:
            retrieval_service = get_retrieval_service()
            for s in result.get("identity_summaries", []):
                emb = s.pop("embedding", None)
                if s.get("crop_base64"):
                    retrieval_service.register_video_identity(
                        global_id=s["global_id"],
                        camera_id=cid,
                        crop_b64=s["crop_base64"],
                        embedding=emb
                    )
        except Exception as reg_err:
            print(f"[Warning] Could not register video identities: {reg_err}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/upload-video")
async def upload_and_process_video(
    file: UploadFile = File(...),
    max_frames: int = Form(200),
    camera_id: int = Form(1)
):
    """Upload an MP4 video file and immediately process it with YOLOv8 + Re-ID."""
    upload_dir = os.path.join(RESULTS_DIR, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    saved_path = os.path.join(upload_dir, file.filename)

    with open(saved_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    service = get_inference_service()
    base_name = os.path.splitext(file.filename)[0]
    out_video_name = f"{base_name}_annotated.mp4"
    out_video_path = os.path.join(RESULTS_DIR, out_video_name)

    try:
        result = service.process_video(
            video_path=saved_path,
            output_video_path=out_video_path,
            max_frames=max_frames,
            camera_id=camera_id
        )
        result["video_url"] = f"http://127.0.0.1:8001/results/{out_video_name}"

        # Register detected persons into Forensic Search pool & gallery
        try:
            retrieval_service = get_retrieval_service()
            for s in result.get("identity_summaries", []):
                emb = s.pop("embedding", None)
                if s.get("crop_base64"):
                    retrieval_service.register_video_identity(
                        global_id=s["global_id"],
                        camera_id=camera_id,
                        crop_b64=s["crop_base64"],
                        embedding=emb
                    )
        except Exception as reg_err:
            print(f"[Warning] Could not register video identities: {reg_err}")

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/live-frame")
def process_live_webcam_frame(req: LiveFrameRequest):
    """Process a single real-time frame from browser webcam or local camera."""
    service = get_inference_service()
    
    # Decode base64 image
    try:
        raw_b64 = req.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[1]
        img_bytes = base64.b64decode(raw_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image frame")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    try:
        res = service.process_single_frame(frame, camera_id=req.camera_id or 0)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/inference/reset-live")
def reset_live_session():
    """Reset live tracker and global identities."""
    service = get_inference_service()
    service.reset_live_session()
    return {"status": "live session reset"}

# ----------------------------------------------------------------------
# 5️⃣ Forensic Search & Gallery Retrieval Endpoints
# ----------------------------------------------------------------------
_retrieval_service = None

def get_retrieval_service():
    global _retrieval_service
    if _retrieval_service is None:
        from backend.services.retrieval_service import ForensicRetrievalService
        _retrieval_service = ForensicRetrievalService(device="cpu")
    return _retrieval_service

class RetrievalRequest(BaseModel):
    query_filename: Optional[str] = None
    query_b64: Optional[str] = None
    query_cam: Optional[int] = 1
    top_k: Optional[int] = 10

@app.get("/api/retrieval/samples")
def get_retrieval_samples():
    """Returns sample query persons with base64 thumbnail."""
    service = get_retrieval_service()
    return service.get_sample_queries()

@app.post("/api/retrieval/search")
def search_reid_query(req: RetrievalRequest):
    """Executes multi-model quality-gated query search against gallery."""
    service = get_retrieval_service()
    query_path = None
    query_b64 = req.query_b64

    if req.query_filename:
        candidate_path = os.path.join(PROJECT_ROOT, "data", "market1501", "Market-1501-v15.09.15", "query", req.query_filename)
        if os.path.exists(candidate_path):
            query_path = candidate_path
        else:
            # Check if this query is a registered video sighting with a base64 thumbnail
            for q in service._sample_queries:
                if q.get("filename") == req.query_filename and q.get("thumbnail_b64"):
                    query_b64 = q["thumbnail_b64"]
                    break

    return service.search_query(
        query_path=query_path,
        query_b64=query_b64,
        query_cam=req.query_cam or 1,
        top_k=req.top_k or 10
    )

# ----------------------------------------------------------------------
# 6️⃣ Dynamic Benchmark & Experiments Data Endpoints
# ----------------------------------------------------------------------
@app.get("/api/experiments/benchmark-data")
def get_benchmark_experiments():
    """Reads all real experimental results from results/ and structures them for charts."""
    import json
    files = {
        "OSNet Baseline": "osnet_market1501.json",
        "TransReID Baseline": "transreid_market1501.json",
        "CLIP Foundation": "clip_market1501.json",
        "OSNet + TransReID": "osnet_transreid_market1501.json",
        "OSNet + CLIP": "osnet_clip_market1501.json",
        "TransReID + CLIP": "transreid_clip_market1501.json",
        "Fixed 3-Model Fusion": "fixed_fusion_market1501.json",
        "Adaptive Gating Fusion": "adaptive_fusion_market1501.json",
        "Adaptive + Pose": "adaptive_pose_market1501.json",
        "Full M²-ReID Pipeline": "m2_reid_full_market1501.json",
    }
    benchmarks = []
    for label, fname in files.items():
        fpath = os.path.join(RESULTS_DIR, fname)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                m = data.get("metrics", data)
                benchmarks.append({
                    "model": label,
                    "rank1": m.get("rank1", 0.0),
                    "rank5": m.get("rank5", 0.0),
                    "rank10": m.get("rank10", 0.0),
                    "mAP": m.get("mAP", 0.0),
                    "latency_ms": m.get("latency_ms", 0.0),
                    "fps": m.get("fps", 0.0),
                    "file": fname
                })
            except Exception:
                continue

    ablation_path = os.path.join(RESULTS_DIR, "ablation_study.json")
    ablation_data = {}
    if os.path.exists(ablation_path):
        try:
            with open(ablation_path, "r") as f:
                ablation_data = json.load(f)
        except Exception:
            pass

    robustness_path = os.path.join(RESULTS_DIR, "robustness_benchmark.json")
    robustness_data = {}
    if os.path.exists(robustness_path):
        try:
            with open(robustness_path, "r") as f:
                robustness_data = json.load(f)
        except Exception:
            pass

    return {
        "benchmarks": benchmarks,
        "ablation": ablation_data,
        "robustness": robustness_data
    }

