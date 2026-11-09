"use client";
import { useEffect, useRef, useState } from "react";
import { fetchDatasets, fetchCameraTopology, CameraTopology, CameraEdge } from "@/lib/api";

// ── SVG topology graph ────────────────────────────────────────────────────────
function toRadial(count: number, cx: number, cy: number, r: number) {
  return Array.from({ length: count }, (_, i) => {
    const angle = (2 * Math.PI * i) / count - Math.PI / 2;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  });
}

function TopologyGraph({ topo }: { topo: CameraTopology }) {
  const W = 600, H = 400, CX = W / 2, CY = H / 2, R = 150;
  const positions = toRadial(topo.nodes.length, CX, CY, R);
  const nodePos = Object.fromEntries(topo.nodes.map((n, i) => [n, positions[i]]));

  // Only show edges with prob > 0.2 to keep graph readable
  const visibleEdges = topo.edges.filter(e => e.probability > 0.2 && e.source < e.target);

  const maxProb = Math.max(...topo.edges.map(e => e.probability), 0.01);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="topology-svg" style={{ maxHeight: 380 }}>
      <defs>
        <radialGradient id="nodeGrad" cx="50%" cy="30%" r="60%">
          <stop offset="0%" stopColor="#06b6d4" />
          <stop offset="100%" stopColor="#3b82f6" />
        </radialGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <marker id="arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
          <path d="M 0 0 L 6 3 L 0 6 z" fill="rgba(6,182,212,0.5)" />
        </marker>
      </defs>

      {/* Edges */}
      {visibleEdges.map((e, i) => {
        const s = nodePos[e.source], t = nodePos[e.target];
        if (!s || !t) return null;
        const opacity = 0.3 + 0.7 * (e.probability / maxProb);
        const sw = 1 + 2 * (e.probability / maxProb);
        const mx = (s.x + t.x) / 2 + (t.y - s.y) * 0.15;
        const my = (s.y + t.y) / 2 - (t.x - s.x) * 0.15;
        return (
          <g key={i} className="topo-edge">
            <path
              d={`M ${s.x} ${s.y} Q ${mx} ${my} ${t.x} ${t.y}`}
              fill="none"
              stroke={`rgba(6,182,212,${opacity})`}
              strokeWidth={sw}
              markerEnd="url(#arrow)"
            />
            <text
              x={mx} y={my} textAnchor="middle" dominantBaseline="middle"
              fill="rgba(255,255,255,0.5)" fontSize={9}
            >
              {e.probability.toFixed(2)}
            </text>
          </g>
        );
      })}

      {/* Nodes */}
      {topo.nodes.map((n, i) => {
        const { x, y } = positions[i];
        return (
          <g key={n} className="topo-node" filter="url(#glow)">
            <circle cx={x} cy={y} r={26} fill="url(#nodeGrad)" opacity={0.9} />
            <circle cx={x} cy={y} r={26} fill="none" stroke="rgba(6,182,212,0.8)" strokeWidth={1.5} />
            <text x={x} y={y - 1} textAnchor="middle" dominantBaseline="middle"
              fill="white" fontSize={12} fontWeight={700} fontFamily="JetBrains Mono, monospace">
              C{String(n).padStart(2, "0")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ── Edge table ────────────────────────────────────────────────────────────────
function EdgeTable({ edges }: { edges: CameraEdge[] }) {
  const sorted = [...edges].sort((a, b) => b.probability - a.probability).slice(0, 12);
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>From</th>
          <th>To</th>
          <th>Probability</th>
          <th>Mean Transit</th>
          <th>Std Dev</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((e, i) => (
          <tr key={i}>
            <td><span className="mono" style={{ color: "var(--accent-cyan)" }}>C{String(e.source).padStart(2, "0")}</span></td>
            <td><span className="mono" style={{ color: "var(--accent-blue)" }}>C{String(e.target).padStart(2, "0")}</span></td>
            <td>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <div style={{ width: 48, height: 4, borderRadius: 2, background: "var(--border)", overflow: "hidden" }}>
                  <div style={{ width: `${e.probability * 100}%`, height: "100%", background: "var(--accent-cyan)", borderRadius: 2 }} />
                </div>
                <span className="mono" style={{ fontSize: 11 }}>{e.probability.toFixed(4)}</span>
              </div>
            </td>
            <td className="mono" style={{ fontSize: 12 }}>{e.mean_transit_s}s</td>
            <td className="mono" style={{ fontSize: 12 }}>±{e.std_transit_s}s</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

// ── Main ─────────────────────────────────────────────────────────────────────
export default function TopologyPage() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [topo, setTopo] = useState<CameraTopology | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchDatasets().then(ns => {
      setDatasets(ns);
      if (ns.length) setSelected(ns[0]);
    });
  }, []);

  useEffect(() => {
    if (!selected) return;
    setLoading(true);
    setError(null);
    fetchCameraTopology(selected)
      .then(t => { setTopo(t); setLoading(false); })
      .catch(e => { setError(e.message); setLoading(false); });
  }, [selected]);

  return (
    <>
      <header className="page-header">
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700 }}>Camera Topology</h1>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
            Spatial-temporal transition graph · Gaussian travel-time model
          </p>
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {datasets.map(n => (
            <button key={n} onClick={() => setSelected(n)} style={{
              padding: "7px 16px", borderRadius: 8, border: "1px solid",
              borderColor: selected === n ? "var(--accent-cyan)" : "var(--border)",
              background: selected === n ? "rgba(6,182,212,0.12)" : "transparent",
              color: selected === n ? "var(--accent-cyan)" : "var(--text-secondary)",
              cursor: "pointer", fontSize: 12, fontWeight: 600, transition: "all 0.15s",
            }}>{n}</button>
          ))}
        </div>
      </header>

      <main className="page-content animate-in">
        {error && (
          <div style={{ padding: 16, background: "rgba(239,68,68,0.1)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 10, marginBottom: 20, color: "#f87171", fontSize: 13 }}>
            ⚠ {error}
          </div>
        )}

        <div className="grid-2" style={{ marginBottom: 20 }}>
          {/* Graph */}
          <div className="glass-card" style={{ padding: 20 }}>
            <div className="section-title" style={{ marginBottom: 4 }}>Transition Graph</div>
            <div className="section-subtitle" style={{ marginBottom: 16 }}>Edge weight = transition probability · Thickness = strength</div>
            {loading
              ? <div className="skeleton" style={{ height: 340, borderRadius: 10 }} />
              : topo && <TopologyGraph topo={topo} />
            }
          </div>

          {/* Stats */}
          <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
            <div className="grid-2">
              {[
                { label: "Camera Nodes", value: topo?.nodes.length ?? "—", accent: "cyan" },
                { label: "Transitions", value: topo?.edges.length ?? "—", accent: "blue" },
              ].map(({ label, value, accent }) => (
                <div key={label} className={`metric-card ${accent}`}>
                  <div className="metric-label">{label}</div>
                  <div className="metric-value" style={{ color: `var(--accent-${accent})`, fontSize: 28 }}>{value}</div>
                </div>
              ))}
            </div>

            <div className="glass-card" style={{ padding: 20, flex: 1 }}>
              <div className="section-title" style={{ marginBottom: 12 }}>Model: Gaussian Travel Time</div>
              <div style={{ color: "var(--text-secondary)", fontSize: 12, lineHeight: 1.8 }}>
                <p>Each camera pair <span className="mono" style={{ color: "var(--accent-cyan)" }}>Cᵢ → Cⱼ</span> has a parameterized travel-time distribution:</p>
                <div style={{
                  margin: "12px 0", padding: "10px 14px",
                  background: "rgba(6,182,212,0.06)", borderRadius: 8,
                  borderLeft: "3px solid var(--accent-cyan)",
                  fontFamily: "JetBrains Mono, monospace", fontSize: 12,
                  color: "var(--accent-cyan)"
                }}>
                  P(Δt | Cᵢ→Cⱼ) ~ N(μᵢⱼ, σᵢⱼ²)
                </div>
                <p>Calibrated pairs use measured μ and σ. Uncalibrated pairs use a distance-weighted exponential fallback.</p>
              </div>
            </div>
          </div>
        </div>

        {/* Edge table */}
        {topo && (
          <div className="glass-card" style={{ padding: 20 }}>
            <div className="section-title" style={{ marginBottom: 4 }}>Top Transitions by Probability</div>
            <div className="section-subtitle" style={{ marginBottom: 14 }}>Showing top 12 edges · sorted by transition probability</div>
            {loading ? <div className="skeleton" style={{ height: 200 }} /> : <EdgeTable edges={topo.edges} />}
          </div>
        )}
      </main>
    </>
  );
}
