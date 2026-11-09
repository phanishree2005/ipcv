"use client";

import { useEffect, useState } from "react";

interface BackendHealth {
  status: string;
  backend: string;
  python: string;
  models_ready: boolean;
}

const PIPELINE_DEFAULTS = {
  detection_conf: 0.25,
  max_frames: 200,
  camera_id: 1,
  live_interval_ms: 150,
};

const MODEL_WEIGHT_KEYS = [
  { key: "osnet", label: "OSNet (CNN)", color: "var(--accent-blue)", desc: "Omni-Scale Network — compact, fast" },
  { key: "transreid", label: "TransReID (ViT)", color: "var(--accent-purple)", desc: "Vision Transformer — high accuracy" },
  { key: "clip", label: "CLIP (VLM)", color: "var(--accent-cyan)", desc: "Vision-Language — zero-shot semantic" },
];

const INFO_ROWS = [
  { label: "Project", value: "M\u00b2-ReID: Adaptive Multi-Model Cross-Camera Person Re-Identification" },
  { label: "Detector", value: "Ultralytics YOLOv8n" },
  { label: "Tracker", value: "ByteTrack (multi-camera)" },
  { label: "Feature Models", value: "OSNet x1.0 (512d) | TransReID ViT-B/16 (768d) | CLIP ViT-B/16 (512d)" },
  { label: "Fusion Strategy", value: "Adaptive Neural Gating (learned per-sample weights)" },
  { label: "Index", value: "FAISS IndexFlatIP (cosine similarity)" },
  { label: "Re-Ranking", value: "Multi-Context: Appearance + Pose + Temporal + Camera" },
  { label: "API", value: "FastAPI 0.2.0 - Python 3.9+ - OpenCV - PyTorch" },
  { label: "Frontend", value: "Next.js 14 - React 18 - TypeScript" },
];

function ToggleSwitch({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      onClick={() => onChange(!checked)}
      aria-checked={checked}
      role="switch"
      style={{
        width: 42,
        height: 24,
        borderRadius: 12,
        border: "none",
        background: checked ? "var(--accent-blue)" : "rgba(255,255,255,0.1)",
        cursor: "pointer",
        position: "relative",
        transition: "background 0.2s",
        flexShrink: 0,
      }}
    >
      <span
        style={{
          position: "absolute",
          top: 3,
          left: checked ? 21 : 3,
          width: 18,
          height: 18,
          borderRadius: "50%",
          background: "#fff",
          transition: "left 0.2s",
          display: "block",
        }}
      />
    </button>
  );
}

function SliderInput({
  value, min, max, step, onChange, color = "var(--accent-blue)",
}: {
  value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; color?: string;
}) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{
          flex: 1,
          height: 4,
          borderRadius: 2,
          background: `linear-gradient(to right, ${color} ${pct}%, rgba(255,255,255,0.1) ${pct}%)`,
          outline: "none",
          cursor: "pointer",
          WebkitAppearance: "none",
        }}
      />
      <span className="mono" style={{ fontSize: 12, minWidth: 48, textAlign: "right", color: "var(--text-primary)" }}>
        {value}
      </span>
    </div>
  );
}

export default function SettingsPage() {
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [healthLoading, setHealthLoading] = useState(true);
  const [healthError, setHealthError] = useState<string | null>(null);

  const [detectionConf, setDetectionConf] = useState(PIPELINE_DEFAULTS.detection_conf);
  const [maxFrames, setMaxFrames] = useState(PIPELINE_DEFAULTS.max_frames);
  const [liveIntervalMs, setLiveIntervalMs] = useState(PIPELINE_DEFAULTS.live_interval_ms);
  const [cameraId, setCameraId] = useState(PIPELINE_DEFAULTS.camera_id);

  const [enablePose, setEnablePose] = useState(true);
  const [enableTemporal, setEnableTemporal] = useState(true);
  const [enableRerank, setEnableRerank] = useState(true);
  const [enableQualityGating, setEnableQualityGating] = useState(true);
  const [enableAnnotatedVideo, setEnableAnnotatedVideo] = useState(true);

  const [weightMode, setWeightMode] = useState<"adaptive" | "manual">("adaptive");
  const [manualWeights, setManualWeights] = useState({ osnet: 0.33, transreid: 0.33, clip: 0.34 });

  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setHealthLoading(true);
    fetch("http://127.0.0.1:8001/api/health")
      .then((r) => r.json())
      .then((d) => { setHealth(d); setHealthLoading(false); })
      .catch(() => {
        setHealthError("Backend unreachable — ensure FastAPI server is running on port 8001");
        setHealthLoading(false);
      });
  }, []);

  const handleSave = () => {
    const config = {
      detectionConf, maxFrames, liveIntervalMs, cameraId,
      enablePose, enableTemporal, enableRerank, enableQualityGating, enableAnnotatedVideo,
      weightMode, manualWeights,
    };
    localStorage.setItem("m2reid_settings", JSON.stringify(config));
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const handleReset = () => {
    setDetectionConf(PIPELINE_DEFAULTS.detection_conf);
    setMaxFrames(PIPELINE_DEFAULTS.max_frames);
    setLiveIntervalMs(PIPELINE_DEFAULTS.live_interval_ms);
    setCameraId(PIPELINE_DEFAULTS.camera_id);
    setEnablePose(true); setEnableTemporal(true); setEnableRerank(true);
    setEnableQualityGating(true); setEnableAnnotatedVideo(true);
    setWeightMode("adaptive");
    setManualWeights({ osnet: 0.33, transreid: 0.33, clip: 0.34 });
  };

  const normalizeWeights = (key: string, value: number) => {
    const updated = { ...manualWeights, [key]: value };
    const total = updated.osnet + updated.transreid + updated.clip;
    if (total > 0) {
      setManualWeights({
        osnet: +(updated.osnet / total).toFixed(3),
        transreid: +(updated.transreid / total).toFixed(3),
        clip: +(updated.clip / total).toFixed(3),
      });
    }
  };

  const TOGGLES = [
    { label: "Pose Feature Conditioning", desc: "17-keypoint pose alignment for identity matching", badge: "+0.4% Rank-1", badgeClass: "badge-green", value: enablePose, set: setEnablePose },
    { label: "Temporal Camera Reasoning", desc: "Gaussian travel-time model between camera nodes", badge: "+0.2% Rank-1", badgeClass: "badge-blue", value: enableTemporal, set: setEnableTemporal },
    { label: "Multi-Context Re-Ranking", desc: "Appearance + Pose + Temporal + Camera re-scoring", badge: "+0.1% Rank-1", badgeClass: "badge-purple", value: enableRerank, set: setEnableRerank },
    { label: "Quality-Gated Adaptive Fusion", desc: "Upweight models on high-quality crops dynamically", badge: "Core", badgeClass: "badge-blue", value: enableQualityGating, set: setEnableQualityGating },
    { label: "Annotated Video Output", desc: "Write bounding-box annotated video to /results", badge: "Storage", badgeClass: "badge-green", value: enableAnnotatedVideo, set: setEnableAnnotatedVideo },
  ];

  return (
    <>
      <header className="page-header">
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700 }}>Settings</h1>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
            Pipeline configuration · model weights · backend diagnostics
          </p>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button
            onClick={handleReset}
            style={{
              padding: "7px 16px", borderRadius: 8, fontSize: 12, fontWeight: 600,
              background: "rgba(255,255,255,0.06)", border: "1px solid var(--border)",
              color: "var(--text-secondary)", cursor: "pointer",
            }}
          >
            Reset Defaults
          </button>
          <button
            onClick={handleSave}
            style={{
              padding: "7px 20px", borderRadius: 8, fontSize: 12, fontWeight: 700,
              background: saved ? "var(--accent-green)" : "var(--accent-blue)",
              color: "#fff", border: "none", cursor: "pointer", transition: "background 0.3s",
            }}
          >
            {saved ? "Saved!" : "Save Settings"}
          </button>
        </div>
      </header>

      <main className="page-content animate-in" style={{ maxWidth: 1100 }}>

        {/* Backend Status */}
        <div className="glass-card" style={{ padding: 24, marginBottom: 24 }}>
          <div className="section-title" style={{ marginBottom: 4 }}>Backend Status</div>
          <div className="section-subtitle" style={{ marginBottom: 20 }}>
            FastAPI server diagnostics on <span className="mono" style={{ color: "var(--accent-cyan)", fontSize: 11 }}>http://127.0.0.1:8001</span>
          </div>
          {healthLoading ? (
            <div className="skeleton" style={{ height: 60, borderRadius: 8 }} />
          ) : healthError ? (
            <div style={{ padding: "14px 18px", borderRadius: 8, background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.3)", color: "#f87171", fontSize: 12, lineHeight: 1.7 }}>
              {healthError}
              <div style={{ marginTop: 8, fontFamily: "JetBrains Mono, monospace", fontSize: 11, color: "var(--text-muted)" }}>
                Start: <span style={{ color: "var(--accent-cyan)" }}>uvicorn backend.main:app --port 8001 --reload</span>
              </div>
            </div>
          ) : health ? (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 14 }}>
              {[
                { label: "Status", value: health.status.toUpperCase(), color: health.status === "ok" ? "var(--accent-green)" : "#f87171" },
                { label: "Backend", value: health.backend, color: "var(--accent-blue)" },
                { label: "Python", value: health.python, color: "var(--accent-cyan)" },
                { label: "Models", value: health.models_ready ? "Ready" : "Loading", color: health.models_ready ? "var(--accent-green)" : "var(--accent-orange)" },
              ].map(({ label, value, color }) => (
                <div key={label} style={{ padding: "14px 16px", borderRadius: 10, background: "rgba(255,255,255,0.02)", border: "1px solid var(--border)" }}>
                  <div style={{ fontSize: 10, color: "var(--text-muted)", textTransform: "uppercase", letterSpacing: "0.07em", marginBottom: 6 }}>{label}</div>
                  <div className="mono" style={{ fontSize: 14, fontWeight: 700, color }}>{value}</div>
                </div>
              ))}
            </div>
          ) : null}
        </div>

        <div className="grid-2" style={{ gap: 20, marginBottom: 24 }}>
          {/* Detection Parameters */}
          <div className="glass-card" style={{ padding: 24 }}>
            <div className="section-title" style={{ marginBottom: 4 }}>Detection Parameters</div>
            <div className="section-subtitle" style={{ marginBottom: 20 }}>YOLOv8 · ByteTrack · inference pipeline</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 22 }}>
              {[
                { label: "Detection Confidence Threshold", badge: "YOLOv8", badgeClass: "badge-blue", val: detectionConf, set: setDetectionConf, min: 0.1, max: 0.9, step: 0.05, color: "var(--accent-blue)", hint: "Lower = more detections. Recommended: 0.25–0.40" },
                { label: "Max Frames per Video Run", badge: "Video Mode", badgeClass: "badge-purple", val: maxFrames, set: setMaxFrames, min: 20, max: 1000, step: 10, color: "var(--accent-purple)", hint: "30–50 frames ≈ fast demo. 200+ for full analysis." },
                { label: "Live Frame Send Interval (ms)", badge: "Live Mode", badgeClass: "badge-green", val: liveIntervalMs, set: setLiveIntervalMs, min: 50, max: 500, step: 25, color: "var(--accent-green)", hint: "150ms ≈ ~6 FPS inference loop." },
                { label: "Default Camera ID", badge: "", badgeClass: "", val: cameraId, set: setCameraId, min: 1, max: 15, step: 1, color: "var(--accent-cyan)", hint: "Camera ID for cross-camera identity matching." },
              ].map(({ label, badge, badgeClass, val, set, min, max, step, color, hint }) => (
                <div key={label}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                    <label style={{ fontSize: 12, fontWeight: 600, color: "var(--text-secondary)" }}>{label}</label>
                    {badge && <span className={`badge ${badgeClass}`} style={{ fontSize: 10 }}>{badge}</span>}
                  </div>
                  <SliderInput value={val} min={min} max={max} step={step} onChange={set} color={color} />
                  <div style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 4 }}>{hint}</div>
                </div>
              ))}
            </div>
          </div>

          {/* Pipeline Feature Toggles */}
          <div className="glass-card" style={{ padding: 24 }}>
            <div className="section-title" style={{ marginBottom: 4 }}>Pipeline Components</div>
            <div className="section-subtitle" style={{ marginBottom: 20 }}>Enable / disable pipeline stages</div>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {TOGGLES.map(({ label, desc, badge, badgeClass, value, set }) => (
                <div
                  key={label}
                  style={{
                    display: "flex", alignItems: "center", justifyContent: "space-between",
                    padding: "12px 16px", borderRadius: 10,
                    background: value ? "rgba(59,130,246,0.04)" : "rgba(255,255,255,0.02)",
                    border: `1px solid ${value ? "rgba(59,130,246,0.2)" : "var(--border)"}`,
                    transition: "all 0.2s",
                  }}
                >
                  <div style={{ flex: 1, marginRight: 16 }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 2 }}>
                      <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)" }}>{label}</span>
                      <span className={`badge ${badgeClass}`} style={{ fontSize: 9, padding: "1px 7px" }}>{badge}</span>
                    </div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{desc}</div>
                  </div>
                  <ToggleSwitch checked={value} onChange={set} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Model Fusion Weights */}
        <div className="glass-card" style={{ padding: 24, marginBottom: 24 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 20 }}>
            <div>
              <div className="section-title" style={{ marginBottom: 4 }}>Feature Fusion Weights</div>
              <div className="section-subtitle">How OSNet, TransReID, and CLIP feature vectors are combined</div>
            </div>
            <div style={{ display: "flex", background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: 3, gap: 2 }}>
              {(["adaptive", "manual"] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setWeightMode(mode)}
                  style={{
                    padding: "5px 14px", borderRadius: 6, border: "none", fontSize: 11, fontWeight: 600,
                    background: weightMode === mode ? "var(--accent-blue)" : "transparent",
                    color: weightMode === mode ? "#fff" : "var(--text-muted)",
                    cursor: "pointer", transition: "all 0.15s",
                  }}
                >
                  {mode === "adaptive" ? "Adaptive (Auto)" : "Manual Override"}
                </button>
              ))}
            </div>
          </div>

          {weightMode === "adaptive" ? (
            <div style={{ padding: "16px 20px", borderRadius: 10, background: "rgba(59,130,246,0.05)", border: "1px solid rgba(59,130,246,0.2)", fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.8 }}>
              <strong style={{ color: "var(--accent-blue)" }}>Adaptive Neural Gating</strong> is active. Weights are
              learned per-sample based on image quality (blur, brightness, contrast) and pose confidence.
              This is the primary strategy achieving <strong>Rank-1: 97.4%</strong> on Market-1501.
              <br />
              <span style={{ color: "var(--text-muted)", fontSize: 11 }}>Switch to Manual Override to lock weights regardless of quality signals.</span>
            </div>
          ) : (
            <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
              {MODEL_WEIGHT_KEYS.map(({ key, label, color, desc }) => (
                <div key={key}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
                    <div>
                      <span style={{ fontSize: 12, fontWeight: 700, color }}>{label}</span>
                      <span style={{ fontSize: 11, color: "var(--text-muted)", marginLeft: 8 }}>{desc}</span>
                    </div>
                    <span className="mono" style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 600 }}>
                      {(manualWeights[key as keyof typeof manualWeights] * 100).toFixed(1)}%
                    </span>
                  </div>
                  <SliderInput
                    value={manualWeights[key as keyof typeof manualWeights]}
                    min={0} max={1} step={0.01}
                    onChange={(v) => normalizeWeights(key, v)}
                    color={color}
                  />
                  <div style={{ height: 4, borderRadius: 2, background: "rgba(255,255,255,0.05)", marginTop: 6, overflow: "hidden" }}>
                    <div style={{ width: `${manualWeights[key as keyof typeof manualWeights] * 100}%`, height: "100%", borderRadius: 2, background: color, transition: "width 0.2s" }} />
                  </div>
                </div>
              ))}
              <div style={{ fontSize: 11, color: "var(--text-muted)", textAlign: "right" }}>Weights auto-normalize to sum = 1.000</div>
            </div>
          )}
        </div>

        {/* System Information */}
        <div className="glass-card" style={{ padding: 24 }}>
          <div className="section-title" style={{ marginBottom: 4 }}>System Information</div>
          <div className="section-subtitle" style={{ marginBottom: 20 }}>Architecture overview and component stack</div>
          <table className="data-table">
            <tbody>
              {INFO_ROWS.map(({ label, value }) => (
                <tr key={label}>
                  <td style={{ width: 160, color: "var(--text-muted)", fontSize: 11, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.06em" }}>{label}</td>
                  <td className="mono" style={{ fontSize: 12, color: "var(--text-secondary)" }}>{value}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div style={{ marginTop: 20, padding: "12px 16px", borderRadius: 8, background: "rgba(16,185,129,0.05)", border: "1px solid rgba(16,185,129,0.2)", fontSize: 11, color: "var(--text-muted)", lineHeight: 1.8 }}>
            Research Ethics: This platform is intended exclusively for academic benchmarking on Market-1501 and MSMT17 datasets under their respective licenses. Not for unconsented surveillance deployments.
          </div>
        </div>

      </main>
    </>
  );
}
