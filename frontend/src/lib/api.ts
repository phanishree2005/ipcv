const BASE = "http://127.0.0.1:8001/api";

export async function fetchHealth() {
  const res = await fetch(`${BASE}/health`);
  if (!res.ok) throw new Error("Backend unreachable");
  return res.json() as Promise<{ status: string; backend: string; python: string; models_ready?: boolean }>;
}

export async function fetchDatasets(): Promise<string[]> {
  const res = await fetch(`${BASE}/datasets`);
  if (!res.ok) throw new Error("Failed to fetch datasets");
  return res.json();
}

export interface DatasetStats {
  dataset: string;
  root_path: string;
  valid: boolean;
  train_images: number;
  query_images: number;
  gallery_images: number;
  train_ids: number;
  query_ids: number;
  gallery_ids: number;
  total_identities: number;
  cameras: number[];
  corrupted_images: number;
  errors?: string[];
}

export async function fetchDatasetStats(name: string): Promise<DatasetStats> {
  const res = await fetch(`${BASE}/datasets/${encodeURIComponent(name)}`);
  if (!res.ok) throw new Error(`Dataset ${name} not found`);
  return res.json();
}

export interface CameraEdge {
  source: string;
  target: string;
  probability: number;
  mean_transit_s: number;
  std_transit_s: number;
}

export interface CameraTopology {
  dataset: string;
  nodes: string[];
  edges: CameraEdge[];
}

export async function fetchCameraTopology(name: string): Promise<CameraTopology> {
  const res = await fetch(`${BASE}/datasets/${encodeURIComponent(name)}/camera-topology`);
  if (!res.ok) throw new Error(`Topology fetch failed for ${name}`);
  return res.json();
}

export async function fetchCameras(name: string): Promise<{ dataset: string; camera_ids: number[] }> {
  const res = await fetch(`${BASE}/datasets/${encodeURIComponent(name)}/cameras`);
  if (!res.ok) throw new Error(`Cameras fetch failed for ${name}`);
  return res.json();
}

// ----------------------------------------------------------------------
// Video & Live Inference API Types & Functions
// ----------------------------------------------------------------------
export interface IdentitySummary {
  global_id: string;
  sighting_count: number;
  cameras_seen: number[];
  first_seen: number;
  last_seen: number;
  confidence: number;
  crop_base64?: string;
}

export interface KeyframeSnapshot {
  frame_idx: number;
  timestamp: number;
  snapshot_b64: string;
}

export interface VideoInferenceResult {
  video_path: string;
  output_video_path?: string;
  video_url?: string;
  total_frames_processed: number;
  total_video_frames: number;
  fps: number;
  resolution: [number, number];
  total_detections: number;
  unique_identities_count: number;
  identity_summaries: IdentitySummary[];
  keyframes: KeyframeSnapshot[];
  frame_sample: Array<{
    frame_idx: number;
    timestamp: number;
    num_detections: number;
    tracked_objects: Array<{
      track_id: number;
      global_id: string;
      bbox: number[];
      confidence: number;
    }>;
  }>;
}

export interface LiveFrameResponse {
  frame_id: number;
  fps: number;
  annotated_frame_base64: string;
  detections_count: number;
  active_tracks: Array<{
    global_id: string;
    track_id: number;
    confidence: number;
    bbox: number[];
  }>;
  identities: IdentitySummary[];
  weights: {
    osnet: number;
    transreid: number;
    clip: number;
  };
}

export async function fetchInferenceSamples(): Promise<{
  sample_videos: Array<{ name: string; path: string; size_mb: number }>;
  results_available: string[];
}> {
  const res = await fetch(`${BASE}/inference/samples`);
  if (!res.ok) throw new Error("Failed to fetch sample videos");
  return res.json();
}

export async function fetchPrecomputedDemo(): Promise<VideoInferenceResult> {
  const res = await fetch(`${BASE}/inference/precomputed-demo`);
  if (!res.ok) throw new Error("Failed to load precomputed demo");
  return res.json();
}

export async function runVideoInference(
  video_path: string,
  max_frames = 200,
  camera_id = 1
): Promise<VideoInferenceResult> {
  const res = await fetch(`${BASE}/inference/run-video`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ video_path, max_frames, camera_id }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Inference failed" }));
    throw new Error(err.detail || "Inference failed");
  }
  return res.json();
}

export async function uploadVideoInference(
  file: File,
  max_frames = 200,
  camera_id = 1
): Promise<VideoInferenceResult> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("max_frames", String(max_frames));
  formData.append("camera_id", String(camera_id));

  const res = await fetch(`${BASE}/inference/upload-video`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Video upload failed" }));
    throw new Error(err.detail || "Video upload failed");
  }
  return res.json();
}

export async function sendLiveFrame(
  image_base64: string,
  camera_id = 0
): Promise<LiveFrameResponse | null> {
  try {
    const res = await fetch(`${BASE}/inference/live-frame`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64, camera_id }),
    });
    if (!res.ok) {
      console.warn("Live frame response status:", res.status);
      return null;
    }
    return res.json();
  } catch (err) {
    console.warn("sendLiveFrame network error:", err);
    return null;
  }
}

export async function resetLiveSession(): Promise<{ status: string }> {
  const res = await fetch(`${BASE}/inference/reset-live`, { method: "POST" });
  return res.json();
}

export interface RetrievalSample {
  filename: string;
  pid: number;
  cam: number;
  seq: number;
  thumbnail_b64: string;
}

export interface RetrievalResultItem {
  rank: number;
  pid: number;
  camera: number;
  filename: string;
  similarity: number;
  composite_score: number;
  is_ground_truth_match: boolean;
  thumbnail_b64: string;
}

export interface RetrievalResponse {
  query: {
    pid: number;
    camera: number;
    image_b64: string;
    weights: { osnet: number; transreid: number; clip: number };
    quality: Record<string, number>;
    pose_keypoints: Array<{ x: number; y: number }>;
  };
  total_gallery_searched: number;
  results: RetrievalResultItem[];
}

export async function fetchRetrievalSamples(): Promise<RetrievalSample[]> {
  const res = await fetch(`${BASE}/retrieval/samples`);
  if (!res.ok) throw new Error("Failed to fetch retrieval samples");
  return res.json();
}

export async function searchRetrievalQuery(params: {
  query_filename?: string;
  query_b64?: string;
  query_cam?: number;
  top_k?: number;
}): Promise<RetrievalResponse> {
  const res = await fetch(`${BASE}/retrieval/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
  if (!res.ok) throw new Error("Search query failed");
  return res.json();
}

export interface BenchmarkItem {
  model: string;
  rank1: number;
  rank5: number;
  rank10: number;
  mAP: number;
  latency_ms: number;
  fps: number;
  file: string;
}

export async function fetchBenchmarkData(): Promise<{
  benchmarks: BenchmarkItem[];
  ablation: Record<string, any>;
  robustness: Record<string, any>;
}> {
  const res = await fetch(`${BASE}/experiments/benchmark-data`);
  if (!res.ok) throw new Error("Failed to fetch benchmark data");
  return res.json();
}

