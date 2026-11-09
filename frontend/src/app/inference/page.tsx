"use client";

import { useEffect, useRef, useState } from "react";
import {
  fetchInferenceSamples,
  fetchPrecomputedDemo,
  runVideoInference,
  uploadVideoInference,
  sendLiveFrame,
  resetLiveSession,
  VideoInferenceResult,
  LiveFrameResponse,
  IdentitySummary
} from "@/lib/api";

const ID_COLORS = [
  "#f87171", // Red
  "#60a5fa", // Blue
  "#4ade80", // Green
  "#facc15", // Yellow
  "#c084fc", // Purple
  "#2dd4bf", // Cyan
  "#fb923c", // Orange
  "#f472b6", // Pink
];

function getColorForId(gid: string): string {
  let hash = 0;
  for (let i = 0; i < gid.length; i++) hash += gid.charCodeAt(i);
  return ID_COLORS[hash % ID_COLORS.length];
}

export default function InferencePage() {
  const [activeTab, setActiveTab] = useState<"video" | "live">("video");

  // --- Video Mode State ---
  const [sampleVideos, setSampleVideos] = useState<Array<{ name: string; path: string; size_mb: number }>>([]);
  const [selectedVideoPath, setSelectedVideoPath] = useState<string>("");
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [maxFrames, setMaxFrames] = useState<number>(40);
  const [cameraId, setCameraId] = useState<number>(1);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [videoResult, setVideoResult] = useState<VideoInferenceResult | null>(null);
  const [videoError, setVideoError] = useState<string | null>(null);
  const [selectedKeyframe, setSelectedKeyframe] = useState<string | null>(null);

  // --- Live Webcam State ---
  const [isLiveActive, setIsLiveActive] = useState<boolean>(false);
  const [liveFps, setLiveFps] = useState<number>(0);
  const [liveViewMode, setLiveViewMode] = useState<"video_overlay" | "server_annotated">("video_overlay");
  const [liveAnnotatedImg, setLiveAnnotatedImg] = useState<string | null>(null);
  const [liveActiveTracks, setLiveActiveTracks] = useState<Array<any>>([]);
  const [liveIdentities, setLiveIdentities] = useState<IdentitySummary[]>([]);
  const [liveWeights, setLiveWeights] = useState({ osnet: 0.33, transreid: 0.33, clip: 0.34 });
  const [liveError, setLiveError] = useState<string | null>(null);
  const [isSendingFrame, setIsSendingFrame] = useState<boolean>(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const liveLoopTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Load samples on mount
  useEffect(() => {
    fetchInferenceSamples()
      .then((data) => {
        setSampleVideos(data.sample_videos || []);
        if (data.sample_videos?.length > 0) {
          const phani = data.sample_videos.find((v) => v.name.includes("phani"));
          setSelectedVideoPath(phani ? phani.path : data.sample_videos[0].path);
        }
      })
      .catch((err) => console.error("Failed to load sample videos:", err));
  }, []);

  // Instant precomputed demo loader
  const handleLoadPrecomputedDemo = async () => {
    setIsProcessing(true);
    setVideoError(null);
    try {
      const res = await fetchPrecomputedDemo();
      setVideoResult(res);
      if (res.keyframes?.length > 0) {
        setSelectedKeyframe(res.keyframes[0].snapshot_b64);
      }
    } catch (err: any) {
      setVideoError(err.message || "Failed to load precomputed demo");
    } finally {
      setIsProcessing(false);
    }
  };

  // Handle Video Pipeline Run
  const handleRunVideo = async () => {
    setIsProcessing(true);
    setVideoError(null);
    setVideoResult(null);

    try {
      let res: VideoInferenceResult;
      if (uploadFile) {
        res = await uploadVideoInference(uploadFile, maxFrames, cameraId);
      } else if (selectedVideoPath) {
        res = await runVideoInference(selectedVideoPath, maxFrames, cameraId);
      } else {
        throw new Error("Please select a sample video or upload an MP4 file.");
      }
      setVideoResult(res);
      if (res.keyframes?.length > 0) {
        setSelectedKeyframe(res.keyframes[0].snapshot_b64);
      }
    } catch (err: any) {
      setVideoError(err.message || "Pipeline inference failed");
    } finally {
      setIsProcessing(false);
    }
  };

  // --- Live Webcam Controls ---
  const startLiveWebcam = async () => {
    setLiveError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
        audio: false,
      });

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current?.play().catch(console.error);
        };
      }

      await resetLiveSession().catch(() => {});
      setIsLiveActive(true);
    } catch (err: any) {
      setLiveError("Webcam access error: " + (err.message || "Permission denied / No device found"));
    }
  };

  const stopLiveWebcam = () => {
    if (liveLoopTimeoutRef.current) {
      clearTimeout(liveLoopTimeoutRef.current);
      liveLoopTimeoutRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsLiveActive(false);
    setLiveAnnotatedImg(null);
    setLiveActiveTracks([]);

    // Clear overlay
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
  };

  const handleResetLive = async () => {
    await resetLiveSession().catch(() => {});
    setLiveIdentities([]);
    setLiveActiveTracks([]);
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext("2d");
      ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
  };

  // Render bounding boxes on overlay canvas
  const renderOverlay = (tracks: Array<any>, frameW: number, frameH: number) => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    canvas.width = frameW;
    canvas.height = frameH;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, frameW, frameH);

    for (const tr of tracks) {
      const [x1, y1, x2, y2] = tr.bbox;
      const color = getColorForId(tr.global_id || "Trk");

      // Draw box
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

      // Label background
      const label = `${tr.global_id} | Trk #${tr.track_id} (${Math.round(tr.confidence * 100)}%)`;
      ctx.font = "bold 13px sans-serif";
      const textWidth = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(x1, Math.max(0, y1 - 22), textWidth + 10, 22);

      ctx.fillStyle = "#000000";
      ctx.fillText(label, x1 + 5, Math.max(16, y1 - 6));
    }
  };

  // Live frame capture loop
  useEffect(() => {
    let isMounted = true;

    const captureAndSend = async () => {
      if (!isLiveActive || !videoRef.current || !captureCanvasRef.current) {
        return;
      }

      const vid = videoRef.current;
      const canvas = captureCanvasRef.current;

      if (vid.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && vid.videoWidth > 0) {
        canvas.width = vid.videoWidth;
        canvas.height = vid.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(vid, 0, 0, canvas.width, canvas.height);
          const dataUrl = canvas.toDataURL("image/jpeg", 0.7);

          try {
            setIsSendingFrame(true);
            const res = await sendLiveFrame(dataUrl, 0);
            if (isMounted && isLiveActive && res) {
              setLiveAnnotatedImg(`data:image/jpeg;base64,${res.annotated_frame_base64}`);
              setLiveFps(res.fps);
              setLiveActiveTracks(res.active_tracks || []);
              setLiveIdentities(res.identities || []);
              if (res.weights) {
                setLiveWeights(res.weights);
              }
              // Draw real-time canvas bounding boxes
              renderOverlay(res.active_tracks || [], vid.videoWidth, vid.videoHeight);
            }
          } catch (e) {
            console.warn("Frame capture error:", e);
          } finally {
            if (isMounted) setIsSendingFrame(false);
          }
        }
      }

      if (isMounted && isLiveActive) {
        liveLoopTimeoutRef.current = setTimeout(captureAndSend, 150);
      }
    };

    if (isLiveActive) {
      captureAndSend();
    }

    return () => {
      isMounted = false;
      if (liveLoopTimeoutRef.current) {
        clearTimeout(liveLoopTimeoutRef.current);
      }
    };
  }, [isLiveActive]);

  return (
    <>
      <header className="page-header">
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700, lineHeight: 1 }}>
            <span className="gradient-text">Live & Video</span> Re-ID Inference Hub
          </h1>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
            Real-time YOLOv8 person detection, ByteTrack tracking, and Adaptive Multi-Model Re-ID (OSNet + TransReID + CLIP)
          </p>
        </div>

        {/* Mode Switcher Tabs */}
        <div style={{ display: "flex", gap: 8, background: "rgba(255,255,255,0.04)", padding: 4, borderRadius: 8 }}>
          <button
            onClick={() => {
              if (isLiveActive) stopLiveWebcam();
              setActiveTab("video");
            }}
            style={{
              padding: "6px 16px",
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 600,
              background: activeTab === "video" ? "var(--accent-blue)" : "transparent",
              color: activeTab === "video" ? "#fff" : "var(--text-muted)",
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s ease"
            }}
          >
            📹 Video File Inference
          </button>
          <button
            onClick={() => setActiveTab("live")}
            style={{
              padding: "6px 16px",
              borderRadius: 6,
              fontSize: 12,
              fontWeight: 600,
              background: activeTab === "live" ? "var(--accent-cyan)" : "transparent",
              color: activeTab === "live" ? "#000" : "var(--text-muted)",
              border: "none",
              cursor: "pointer",
              transition: "all 0.2s ease"
            }}
          >
            🔴 Live Webcam Stream
          </button>
        </div>
      </header>

      <div className="page-content" style={{ padding: "20px 24px", maxWidth: 1400, margin: "0 auto" }}>
        {/* =========================================================================
            TAB 1: VIDEO FILE INFERENCE
            ========================================================================= */}
        {activeTab === "video" && (
          <div style={{ display: "grid", gridTemplateColumns: "360px 1fr", gap: 24 }}>
            {/* Left Column: Video Controls & Input */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Input Card */}
              <div className="panel" style={{ padding: 20, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)" }}>
                <h2 style={{ fontSize: 14, fontWeight: 700, marginBottom: 16, display: "flex", alignItems: "center", gap: 8 }}>
                  <span>📂</span> Video Source
                </h2>

                {/* Option 1: Sample Videos */}
                <div style={{ marginBottom: 16 }}>
                  <label style={{ fontSize: 11, color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                    Select Existing Video on System
                  </label>
                  <select
                    value={selectedVideoPath}
                    onChange={(e) => {
                      setSelectedVideoPath(e.target.value);
                      setUploadFile(null);
                    }}
                    disabled={isProcessing}
                    style={{
                      width: "100%",
                      padding: "8px 12px",
                      borderRadius: 6,
                      background: "rgba(0,0,0,0.3)",
                      border: "1px solid var(--border)",
                      color: "var(--text-primary)",
                      fontSize: 12
                    }}
                  >
                    {sampleVideos.map((s) => (
                      <option key={s.path} value={s.path}>
                        {s.name} ({s.size_mb} MB)
                      </option>
                    ))}
                  </select>
                </div>

                <div style={{ textAlign: "center", margin: "10px 0", fontSize: 11, color: "var(--text-muted)" }}>
                  — OR —
                </div>

                {/* Option 2: Upload File */}
                <div style={{ marginBottom: 16 }}>
                  <label style={{ fontSize: 11, color: "var(--text-muted)", display: "block", marginBottom: 6 }}>
                    Upload New MP4 Video
                  </label>
                  <input
                    type="file"
                    accept="video/mp4,video/x-m4v,video/*"
                    onChange={(e) => {
                      if (e.target.files?.[0]) {
                        setUploadFile(e.target.files[0]);
                        setSelectedVideoPath("");
                      }
                    }}
                    disabled={isProcessing}
                    style={{
                      width: "100%",
                      padding: "8px",
                      borderRadius: 6,
                      background: "rgba(0,0,0,0.3)",
                      border: "1px dashed var(--border)",
                      color: "var(--text-primary)",
                      fontSize: 11
                    }}
                  />
                  {uploadFile && (
                    <div style={{ fontSize: 10, color: "var(--accent-cyan)", marginTop: 4 }}>
                      Selected: {uploadFile.name} ({(uploadFile.size / (1024 * 1024)).toFixed(1)} MB)
                    </div>
                  )}
                </div>

                {/* Parameters */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 20 }}>
                  <div>
                    <label style={{ fontSize: 11, color: "var(--text-muted)", display: "block", marginBottom: 4 }}>
                      Max Frames
                    </label>
                    <input
                      type="number"
                      value={maxFrames}
                      onChange={(e) => setMaxFrames(Number(e.target.value))}
                      disabled={isProcessing}
                      min={20}
                      max={1000}
                      style={{
                        width: "100%",
                        padding: "6px 10px",
                        borderRadius: 6,
                        background: "rgba(0,0,0,0.3)",
                        border: "1px solid var(--border)",
                        color: "var(--text-primary)",
                        fontSize: 12
                      }}
                    />
                  </div>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
                      <label style={{ fontSize: 11, color: "var(--text-muted)" }}>
                        Camera ID
                      </label>
                      <div style={{ display: "flex", gap: 4 }}>
                        <button
                          type="button"
                          onClick={() => setCameraId(1)}
                          style={{
                            padding: "2px 6px",
                            fontSize: 10,
                            borderRadius: 4,
                            background: cameraId === 1 ? "var(--accent-blue)" : "rgba(255,255,255,0.1)",
                            color: "#fff",
                            border: "none",
                            cursor: "pointer"
                          }}
                        >
                          Cam 1
                        </button>
                        <button
                          type="button"
                          onClick={() => setCameraId(2)}
                          style={{
                            padding: "2px 6px",
                            fontSize: 10,
                            borderRadius: 4,
                            background: cameraId === 2 ? "var(--accent-cyan)" : "rgba(255,255,255,0.1)",
                            color: "#fff",
                            border: "none",
                            cursor: "pointer"
                          }}
                        >
                          Cam 2
                        </button>
                      </div>
                    </div>
                    <input
                      type="number"
                      value={cameraId}
                      onChange={(e) => setCameraId(Number(e.target.value))}
                      disabled={isProcessing}
                      min={1}
                      max={15}
                      style={{
                        width: "100%",
                        padding: "6px 10px",
                        borderRadius: 6,
                        background: "rgba(0,0,0,0.3)",
                        border: "1px solid var(--border)",
                        color: "var(--text-primary)",
                        fontSize: 12
                      }}
                    />
                  </div>
                </div>

                {/* Instant Verified Demo Button */}
                <button
                  onClick={handleLoadPrecomputedDemo}
                  disabled={isProcessing}
                  style={{
                    width: "100%",
                    padding: "11px",
                    borderRadius: 8,
                    background: "rgba(6, 182, 212, 0.15)",
                    color: "var(--accent-cyan)",
                    fontWeight: 700,
                    fontSize: 12,
                    border: "1px solid rgba(6, 182, 212, 0.4)",
                    cursor: isProcessing ? "not-allowed" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                    marginBottom: 10,
                  }}
                >
                  ⚡ Instant Verified Demo (phani.mp4 · 195 Frames)
                </button>

                {/* Action Button */}
                <button
                  onClick={handleRunVideo}
                  disabled={isProcessing || (!selectedVideoPath && !uploadFile)}
                  style={{
                    width: "100%",
                    padding: "12px",
                    borderRadius: 8,
                    background: isProcessing
                      ? "rgba(255,255,255,0.1)"
                      : "linear-gradient(135deg, var(--accent-blue), #2563eb)",
                    color: "#fff",
                    fontWeight: 700,
                    fontSize: 13,
                    border: "none",
                    cursor: isProcessing ? "not-allowed" : "pointer",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 8,
                    boxShadow: isProcessing ? "none" : "0 4px 14px rgba(37,99,235,0.3)"
                  }}
                >
                  {isProcessing ? (
                    <>
                      <span className="pulse-dot" /> Running YOLOv8 + Re-ID Pipeline...
                    </>
                  ) : (
                    <>🚀 Execute Re-ID Pipeline</>
                  )}
                </button>

                <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 8, textAlign: "center", lineHeight: 1.4 }}>
                  💡 Tip: Set Max Frames to 30–50 for fast CPU inference (~5s), or click Instant Demo above for full 195 frames.
                </div>

                {videoError && (
                  <div style={{ marginTop: 14, padding: 10, borderRadius: 6, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#f87171", fontSize: 11 }}>
                    ⚠️ {videoError}
                  </div>
                )}
              </div>

              {/* Pipeline Architecture Info */}
              <div className="panel" style={{ padding: 16, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-muted)", textTransform: "uppercase", marginBottom: 10 }}>
                  Pipeline Stack
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 11 }}>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Detector:</span>
                    <span style={{ fontWeight: 600, color: "var(--accent-blue)" }}>Ultralytics YOLOv8</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Tracker:</span>
                    <span style={{ fontWeight: 600, color: "var(--accent-cyan)" }}>ByteTrack</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>Feature Extractor:</span>
                    <span style={{ fontWeight: 600, color: "var(--accent-purple)" }}>OSNet + TransReID + CLIP</span>
                  </div>
                  <div style={{ display: "flex", justifyContent: "space-between" }}>
                    <span style={{ color: "var(--text-muted)" }}>ID Association:</span>
                    <span style={{ fontWeight: 600, color: "var(--accent-green)" }}>Global Identity Manager</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Column: Inference Results & Interactive Player */}
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {videoResult ? (
                <>
                  {/* Summary Metric Cards */}
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
                    <div className="metric-card blue">
                      <div className="metric-label">Frames Processed</div>
                      <div className="metric-value" style={{ color: "var(--accent-blue)" }}>
                        {videoResult.total_frames_processed}
                      </div>
                      <div className="metric-sub">Total {videoResult.total_video_frames} in video</div>
                    </div>

                    <div className="metric-card cyan">
                      <div className="metric-label">Person Detections</div>
                      <div className="metric-value" style={{ color: "var(--accent-cyan)" }}>
                        {videoResult.total_detections}
                      </div>
                      <div className="metric-sub">Real YOLOv8 bboxes</div>
                    </div>

                    <div className="metric-card purple">
                      <div className="metric-label">Unique Global IDs</div>
                      <div className="metric-value" style={{ color: "var(--accent-purple)" }}>
                        {videoResult.unique_identities_count}
                      </div>
                      <div className="metric-sub">Across tracklets</div>
                    </div>

                    <div className="metric-card green">
                      <div className="metric-label">Video FPS / Resolution</div>
                      <div className="metric-value" style={{ color: "var(--accent-green)", fontSize: 18 }}>
                        {videoResult.resolution?.[0]}×{videoResult.resolution?.[1]}
                      </div>
                      <div className="metric-sub">{videoResult.fps} FPS</div>
                    </div>
                  </div>

                  {/* Main Annotated Video & Snapshot Viewer */}
                  <div className="panel" style={{ padding: 20, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
                      <h2 style={{ fontSize: 14, fontWeight: 700, margin: 0 }}>
                        🎬 Annotated Video & Detection Highlights
                      </h2>
                      {videoResult.video_url && (
                        <a
                          href={videoResult.video_url}
                          download
                          className="badge badge-blue"
                          style={{ textDecoration: "none", cursor: "pointer" }}
                        >
                          ⬇ Download Output MP4
                        </a>
                      )}
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 16 }}>
                      {/* Video Player or Keyframe preview */}
                      <div style={{ borderRadius: 8, overflow: "hidden", background: "#000", position: "relative", minHeight: 340, display: "flex", alignItems: "center", justifyContent: "center" }}>
                        {selectedKeyframe ? (
                          <img
                            src={`data:image/jpeg;base64,${selectedKeyframe}`}
                            alt="Annotated Frame"
                            style={{ width: "100%", height: "auto", maxHeight: 400, objectFit: "contain" }}
                          />
                        ) : videoResult.video_url ? (
                          <video
                            controls
                            autoPlay
                            loop
                            style={{ width: "100%", height: "auto", maxHeight: 400 }}
                            src={videoResult.video_url}
                          />
                        ) : (
                          <div style={{ color: "var(--text-muted)", fontSize: 12 }}>No preview available</div>
                        )}
                      </div>

                      {/* Filmstrip Keyframes */}
                      <div>
                        <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-muted)", marginBottom: 8 }}>
                          Keyframe Snapshots ({videoResult.keyframes?.length || 0})
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: 8, maxHeight: 350, overflowY: "auto", paddingRight: 4 }}>
                          {videoResult.keyframes?.map((kf, i) => (
                            <div
                              key={i}
                              onClick={() => setSelectedKeyframe(kf.snapshot_b64)}
                              style={{
                                display: "flex",
                                alignItems: "center",
                                gap: 10,
                                padding: 6,
                                borderRadius: 6,
                                background: selectedKeyframe === kf.snapshot_b64 ? "rgba(59,130,246,0.2)" : "rgba(255,255,255,0.03)",
                                border: selectedKeyframe === kf.snapshot_b64 ? "1px solid var(--accent-blue)" : "1px solid transparent",
                                cursor: "pointer"
                              }}
                            >
                              <img
                                src={`data:image/jpeg;base64,${kf.snapshot_b64}`}
                                alt={`Frame ${kf.frame_idx}`}
                                style={{ width: 64, height: 40, objectFit: "cover", borderRadius: 4 }}
                              />
                              <div>
                                <div style={{ fontSize: 11, fontWeight: 600, color: "var(--text-primary)" }}>
                                  Frame #{kf.frame_idx}
                                </div>
                                <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                                  Time: {kf.timestamp}s
                                </div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Discovered Global Identities Gallery */}
                  <div className="panel" style={{ padding: 20, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)" }}>
                    <h2 style={{ fontSize: 14, fontWeight: 700, marginBottom: 14 }}>
                      👤 Discovered Global Identities ({videoResult.identity_summaries?.length || 0})
                    </h2>

                    <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(220px, 1fr))", gap: 14 }}>
                      {videoResult.identity_summaries?.map((idSummary) => (
                        <div
                          key={idSummary.global_id}
                          style={{
                            borderRadius: 8,
                            padding: 12,
                            background: "rgba(255,255,255,0.02)",
                            border: "1px solid var(--border)",
                            display: "flex",
                            flexDirection: "column",
                            gap: 10
                          }}
                        >
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span className="badge badge-purple" style={{ fontSize: 12, fontWeight: 700 }}>
                              {idSummary.global_id}
                            </span>
                            <span style={{ fontSize: 10, color: "var(--accent-green)", fontWeight: 600 }}>
                              {(idSummary.confidence * 100).toFixed(0)}% Conf
                            </span>
                          </div>

                          {idSummary.crop_base64 && (
                            <div style={{ width: "100%", height: 140, borderRadius: 6, overflow: "hidden", background: "#05070e", display: "flex", alignItems: "center", justifyContent: "center" }}>
                              <img
                                src={`data:image/jpeg;base64,${idSummary.crop_base64}`}
                                alt={idSummary.global_id}
                                style={{ height: "100%", width: "auto", objectFit: "contain" }}
                              />
                            </div>
                          )}

                          <div style={{ fontSize: 11, display: "flex", flexDirection: "column", gap: 4 }}>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: "var(--text-muted)" }}>Sightings:</span>
                              <span style={{ fontWeight: 600 }}>{idSummary.sighting_count} frames</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: "var(--text-muted)" }}>Camera:</span>
                              <span style={{ fontWeight: 600 }}>Cam #{idSummary.cameras_seen.join(", ")}</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                              <span style={{ color: "var(--text-muted)" }}>Timeline:</span>
                              <span style={{ fontWeight: 600 }}>{idSummary.first_seen}s - {idSummary.last_seen}s</span>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </>
              ) : (
                <div
                  style={{
                    minHeight: 450,
                    borderRadius: 12,
                    border: "2px dashed var(--border)",
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    justifyContent: "center",
                    color: "var(--text-muted)",
                    padding: 30,
                    textAlign: "center"
                  }}
                >
                  <div style={{ fontSize: 48, marginBottom: 12 }}>📹</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: "var(--text-primary)", marginBottom: 6 }}>
                    No Video Processed Yet
                  </div>
                  <div style={{ fontSize: 12, maxWidth: 420 }}>
                    Select a sample video from the left panel (e.g. <code>phani.mp4</code>) or upload your own MP4, then click <strong>Execute Re-ID Pipeline</strong> to run YOLOv8 detection and identity tracking.
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* =========================================================================
            TAB 2: LIVE WEBCAM REAL-TIME RE-ID
            ========================================================================= */}
        {activeTab === "live" && (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: 24 }}>
            {/* Left: Video Feed & Real-time Annotated Canvas */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              <div className="panel" style={{ padding: 20, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <h2 style={{ fontSize: 15, fontWeight: 700, margin: 0 }}>
                      🔴 Live Camera Stream & Detection HUD
                    </h2>
                    {isLiveActive && (
                      <span className="badge badge-green">
                        <span className="pulse-dot" /> LIVE ({liveFps} FPS)
                      </span>
                    )}
                  </div>

                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    {isLiveActive && (
                      <div style={{ display: "flex", background: "rgba(255,255,255,0.06)", borderRadius: 6, padding: 2 }}>
                        <button
                          onClick={() => setLiveViewMode("video_overlay")}
                          style={{
                            padding: "4px 10px",
                            fontSize: 11,
                            borderRadius: 4,
                            border: "none",
                            background: liveViewMode === "video_overlay" ? "var(--accent-blue)" : "transparent",
                            color: "#fff",
                            cursor: "pointer"
                          }}
                        >
                          Camera + HUD
                        </button>
                        <button
                          onClick={() => setLiveViewMode("server_annotated")}
                          style={{
                            padding: "4px 10px",
                            fontSize: 11,
                            borderRadius: 4,
                            border: "none",
                            background: liveViewMode === "server_annotated" ? "var(--accent-blue)" : "transparent",
                            color: "#fff",
                            cursor: "pointer"
                          }}
                        >
                          Server Feed
                        </button>
                      </div>
                    )}

                    {!isLiveActive ? (
                      <button
                        onClick={startLiveWebcam}
                        style={{
                          padding: "8px 18px",
                          borderRadius: 6,
                          background: "var(--accent-green)",
                          color: "#000",
                          fontWeight: 700,
                          fontSize: 12,
                          border: "none",
                          cursor: "pointer"
                        }}
                      >
                        ▶ Start Webcam
                      </button>
                    ) : (
                      <>
                        <button
                          onClick={handleResetLive}
                          style={{
                            padding: "8px 14px",
                            borderRadius: 6,
                            background: "rgba(255,255,255,0.1)",
                            color: "var(--text-primary)",
                            fontSize: 12,
                            border: "none",
                            cursor: "pointer"
                          }}
                        >
                          🔄 Reset IDs
                        </button>
                        <button
                          onClick={stopLiveWebcam}
                          style={{
                            padding: "8px 18px",
                            borderRadius: 6,
                            background: "#ef4444",
                            color: "#fff",
                            fontWeight: 700,
                            fontSize: 12,
                            border: "none",
                            cursor: "pointer"
                          }}
                        >
                          ⏹ Stop Camera
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {liveError && (
                  <div style={{ marginBottom: 14, padding: 10, borderRadius: 6, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", color: "#f87171", fontSize: 12 }}>
                    ⚠️ {liveError}
                  </div>
                )}

                {/* Viewport Frame */}
                <div
                  style={{
                    width: "100%",
                    minHeight: 460,
                    borderRadius: 10,
                    overflow: "hidden",
                    background: "#080b12",
                    position: "relative",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    border: "1px solid var(--border)"
                  }}
                >
                  {/* Invisible capture canvas */}
                  <canvas ref={captureCanvasRef} style={{ display: "none" }} />

                  {/* 1. Live hardware camera video element */}
                  <video
                    ref={videoRef}
                    playsInline
                    muted
                    autoPlay
                    style={{
                      width: "100%",
                      height: "auto",
                      maxHeight: 520,
                      objectFit: "contain",
                      display: isLiveActive && liveViewMode === "video_overlay" ? "block" : "none",
                    }}
                  />

                  {/* 2. Real-time Bounding Box Overlay Canvas on top of video */}
                  <canvas
                    ref={overlayCanvasRef}
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: "100%",
                      pointerEvents: "none",
                      objectFit: "contain",
                      display: isLiveActive && liveViewMode === "video_overlay" ? "block" : "none",
                    }}
                  />

                  {/* 3. Server-side annotated image fallback */}
                  {isLiveActive && liveViewMode === "server_annotated" && (
                    liveAnnotatedImg ? (
                      <img
                        src={liveAnnotatedImg}
                        alt="Server Annotated Stream"
                        style={{ width: "100%", height: "auto", maxHeight: 520, objectFit: "contain" }}
                      />
                    ) : (
                      <div style={{ color: "var(--text-muted)", fontSize: 12 }}>
                        <span className="pulse-dot" /> Waiting for first server detection frame...
                      </div>
                    )
                  )}

                  {/* 4. Inactive placeholder */}
                  {!isLiveActive && (
                    <div style={{ textAlign: "center", color: "var(--text-muted)", padding: 40 }}>
                      <div style={{ fontSize: 44, marginBottom: 10 }}>📷</div>
                      <div style={{ fontSize: 14, fontWeight: 600, color: "var(--text-primary)", marginBottom: 4 }}>
                        Webcam Inactive
                      </div>
                      <div style={{ fontSize: 12, maxWidth: 360, margin: "0 auto" }}>
                        Click <strong>Start Webcam</strong> to initiate real-time browser camera capture and live YOLOv8 person tracking.
                      </div>
                    </div>
                  )}
                </div>
              </div>

              {/* Dynamic Model Weights Visualizer */}
              <div className="panel" style={{ padding: 16, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 10, display: "flex", justifyContent: "space-between" }}>
                  <span>🧠 Adaptive Multi-Model Fusion Weights</span>
                  <span style={{ color: "var(--text-muted)", fontSize: 10 }}>Auto-tuned per crop quality & pose</span>
                </div>

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14 }}>
                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                      <span style={{ color: "var(--accent-blue)", fontWeight: 600 }}>OSNet (CNN)</span>
                      <span>{(liveWeights.osnet * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.1)", overflow: "hidden" }}>
                      <div style={{ width: `${liveWeights.osnet * 100}%`, height: "100%", background: "var(--accent-blue)", transition: "width 0.3s ease" }} />
                    </div>
                  </div>

                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                      <span style={{ color: "var(--accent-purple)", fontWeight: 600 }}>TransReID</span>
                      <span>{(liveWeights.transreid * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.1)", overflow: "hidden" }}>
                      <div style={{ width: `${liveWeights.transreid * 100}%`, height: "100%", background: "var(--accent-purple)", transition: "width 0.3s ease" }} />
                    </div>
                  </div>

                  <div>
                    <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 4 }}>
                      <span style={{ color: "var(--accent-cyan)", fontWeight: 600 }}>CLIP ViT-B/16</span>
                      <span>{(liveWeights.clip * 100).toFixed(1)}%</span>
                    </div>
                    <div style={{ height: 6, borderRadius: 3, background: "rgba(255,255,255,0.1)", overflow: "hidden" }}>
                      <div style={{ width: `${liveWeights.clip * 100}%`, height: "100%", background: "var(--accent-cyan)", transition: "width 0.3s ease" }} />
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Right: Live Telemetry & Active Identities Feed */}
            <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
              {/* Telemetry Panel */}
              <div className="panel" style={{ padding: 16, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)" }}>
                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 12 }}>
                  📊 Real-time Telemetry
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10 }}>
                  <div style={{ padding: 10, borderRadius: 8, background: "rgba(255,255,255,0.03)" }}>
                    <div style={{ fontSize: 10, color: "var(--text-muted)" }}>Active Tracks</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent-cyan)" }}>
                      {liveActiveTracks.length}
                    </div>
                  </div>
                  <div style={{ padding: 10, borderRadius: 8, background: "rgba(255,255,255,0.03)" }}>
                    <div style={{ fontSize: 10, color: "var(--text-muted)" }}>Total Global IDs</div>
                    <div style={{ fontSize: 18, fontWeight: 700, color: "var(--accent-purple)" }}>
                      {liveIdentities.length}
                    </div>
                  </div>
                </div>
              </div>

              {/* Real-time Identified Persons */}
              <div className="panel" style={{ padding: 16, borderRadius: 12, background: "rgba(20,24,35,0.7)", border: "1px solid var(--border)", flex: 1, display: "flex", flexDirection: "column" }}>
                <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 12 }}>
                  👤 Discovered Live Identities ({liveIdentities.length})
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: 10, overflowY: "auto", maxHeight: 420 }}>
                  {liveIdentities.length > 0 ? (
                    liveIdentities.map((id) => (
                      <div
                        key={id.global_id}
                        style={{
                          display: "flex",
                          alignItems: "center",
                          gap: 12,
                          padding: 10,
                          borderRadius: 8,
                          background: "rgba(255,255,255,0.03)",
                          border: "1px solid var(--border)"
                        }}
                      >
                        {id.crop_base64 ? (
                          <img
                            src={`data:image/jpeg;base64,${id.crop_base64}`}
                            alt={id.global_id}
                            style={{ width: 44, height: 56, objectFit: "cover", borderRadius: 4 }}
                          />
                        ) : (
                          <div style={{ width: 44, height: 56, borderRadius: 4, background: "rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18 }}>
                            👤
                          </div>
                        )}
                        <div style={{ flex: 1 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span style={{ fontSize: 12, fontWeight: 700, color: "var(--accent-purple)" }}>
                              {id.global_id}
                            </span>
                            <span style={{ fontSize: 10, color: "var(--accent-green)" }}>
                              {(id.confidence * 100).toFixed(0)}%
                            </span>
                          </div>
                          <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 2 }}>
                            Seen: {id.sighting_count} times
                          </div>
                          <div style={{ fontSize: 10, color: "var(--text-muted)" }}>
                            Last seen: {id.last_seen}s
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <div style={{ textAlign: "center", color: "var(--text-muted)", fontSize: 11, padding: "30px 10px" }}>
                      No persons identified yet. Point the camera at a person to detect and register identity.
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
