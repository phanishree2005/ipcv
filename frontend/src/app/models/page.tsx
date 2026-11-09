"use client";

const MODELS = [
  {
    name: "OSNet",
    fullName: "Omni-Scale Network",
    type: "CNN",
    params: "2.2M",
    input: "256×128",
    backbone: "OSNet × 1.0",
    features: 512,
    role: "Lightweight real-time Re-ID backbone. Aggregates omni-scale features via unified aggregation gates (AG).",
    strengths: ["Fast inference", "Compact (2.2M params)", "Good on occluded persons"],
    color: "var(--accent-blue)",
    badge: "badge-blue",
  },
  {
    name: "TransReID",
    fullName: "Transformer-based Re-ID",
    type: "Transformer",
    params: "86M",
    input: "256×128",
    backbone: "ViT-Base/16",
    features: 768,
    role: "Vision Transformer adapted for Re-ID. Uses Side Information Embedding (SIE) for camera and viewpoint tokens.",
    strengths: ["Strong global context", "Camera-aware embedding", "SOTA on Market-1501"],
    color: "var(--accent-purple)",
    badge: "badge-purple",
  },
  {
    name: "CLIP",
    fullName: "Contrastive Language-Image Pre-training",
    type: "Vision-Language",
    params: "149M",
    input: "224×224",
    backbone: "ViT-B/16",
    features: 512,
    role: "Zero-shot visual encoder. Used as a complementary feature extractor for cross-modal and semantic identity matching.",
    strengths: ["Rich semantic features", "Zero-shot generalization", "Language-guided retrieval"],
    color: "var(--accent-cyan)",
    badge: "badge-green",
  },
];

const FUSION = [
  { name: "Adaptive Weighted Fusion", description: "Learns per-sample weights via a quality-gating network. Upweights high-confidence model outputs.", type: "adaptive" },
  { name: "Fixed Baseline Fusion", description: "Uniform average of all model feature vectors. Used as ablation baseline.", type: "fixed" },
];

export default function ModelsPage() {
  return (
    <>
      <header className="page-header">
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700 }}>Model Architecture</h1>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
            Ensemble Re-ID stack — OSNet · TransReID · CLIP
          </p>
        </div>
        <span className="badge badge-green">3 Models Active</span>
      </header>

      <main className="page-content animate-in">
        {/* Model cards */}
        <div className="grid-3" style={{ marginBottom: 24 }}>
          {MODELS.map(m => (
            <div key={m.name} className="glass-card" style={{ padding: 24, borderColor: `${m.color}30`, transition: "all 0.2s" }}>
              {/* Header */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 14 }}>
                <div>
                  <div style={{ fontSize: 20, fontWeight: 800, color: m.color }}>{m.name}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{m.fullName}</div>
                </div>
                <span className={`badge ${m.badge}`}>{m.type}</span>
              </div>

              {/* Specs */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
                {[
                  { k: "Parameters", v: m.params },
                  { k: "Input Size", v: m.input },
                  { k: "Backbone", v: m.backbone },
                  { k: "Feature Dim", v: `${m.features}D` },
                ].map(({ k, v }) => (
                  <div key={k} style={{
                    padding: "8px 10px",
                    background: "rgba(255,255,255,0.02)",
                    borderRadius: 6, border: "1px solid var(--border)"
                  }}>
                    <div style={{ fontSize: 10, color: "var(--text-muted)", marginBottom: 2 }}>{k}</div>
                    <div className="mono" style={{ fontSize: 12, color: "var(--text-primary)", fontWeight: 600 }}>{v}</div>
                  </div>
                ))}
              </div>

              {/* Role */}
              <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.7, marginBottom: 12 }}>{m.role}</p>

              {/* Strengths */}
              <div>
                <div style={{ fontSize: 10, fontWeight: 600, color: "var(--text-muted)", letterSpacing: "0.06em", textTransform: "uppercase", marginBottom: 6 }}>Strengths</div>
                {m.strengths.map(s => (
                  <div key={s} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4, fontSize: 12, color: "var(--text-secondary)" }}>
                    <span style={{ color: m.color, fontSize: 10 }}>▸</span> {s}
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        {/* Fusion strategies */}
        <div className="glass-card" style={{ padding: 24, marginBottom: 24 }}>
          <div className="section-title" style={{ marginBottom: 4 }}>Feature Fusion Strategies</div>
          <div className="section-subtitle" style={{ marginBottom: 18 }}>How model outputs are combined for final identity matching</div>
          <div className="grid-2">
            {FUSION.map(f => (
              <div key={f.name} style={{
                padding: "16px 20px", borderRadius: 10,
                background: f.type === "adaptive" ? "rgba(59,130,246,0.06)" : "rgba(255,255,255,0.02)",
                border: `1px solid ${f.type === "adaptive" ? "rgba(59,130,246,0.3)" : "var(--border)"}`,
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 8 }}>
                  <div style={{
                    width: 8, height: 8, borderRadius: "50%",
                    background: f.type === "adaptive" ? "var(--accent-blue)" : "var(--text-muted)"
                  }} />
                  <div style={{ fontWeight: 700, fontSize: 13, color: f.type === "adaptive" ? "var(--accent-blue)" : "var(--text-primary)" }}>
                    {f.name}
                  </div>
                  {f.type === "adaptive" && <span className="badge badge-blue" style={{ fontSize: 10 }}>Primary</span>}
                </div>
                <p style={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.7 }}>{f.description}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Pipeline diagram */}
        <div className="glass-card" style={{ padding: 24 }}>
          <div className="section-title" style={{ marginBottom: 16 }}>Inference Pipeline</div>
          <div style={{ display: "flex", alignItems: "center", gap: 0, overflowX: "auto", paddingBottom: 8 }}>
            {[
              { step: "Input Frame", color: "var(--text-muted)", bg: "rgba(255,255,255,0.04)" },
              { step: "YOLOv8 Detection", color: "var(--accent-orange)", bg: "rgba(245,158,11,0.1)" },
              { step: "Person Crops", color: "var(--text-muted)", bg: "rgba(255,255,255,0.04)" },
              { step: "OSNet + TransReID + CLIP", color: "var(--accent-blue)", bg: "rgba(59,130,246,0.1)" },
              { step: "Adaptive Fusion", color: "var(--accent-purple)", bg: "rgba(139,92,246,0.1)" },
              { step: "FAISS Retrieval", color: "var(--accent-cyan)", bg: "rgba(6,182,212,0.1)" },
              { step: "Re-ranking", color: "var(--accent-green)", bg: "rgba(16,185,129,0.1)" },
              { step: "Identity Match", color: "var(--accent-green)", bg: "rgba(16,185,129,0.15)" },
            ].map((s, i, arr) => (
              <div key={i} style={{ display: "flex", alignItems: "center", flexShrink: 0 }}>
                <div style={{
                  padding: "8px 14px", borderRadius: 8, background: s.bg,
                  border: `1px solid ${s.color}40`, color: s.color,
                  fontSize: 11, fontWeight: 600, whiteSpace: "nowrap",
                }}>
                  {s.step}
                </div>
                {i < arr.length - 1 && (
                  <div style={{ padding: "0 6px", color: "var(--text-muted)", fontSize: 16 }}>→</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </main>
    </>
  );
}
