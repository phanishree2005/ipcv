"use client";
import { useEffect, useState } from "react";
import { fetchDatasets, fetchDatasetStats, DatasetStats } from "@/lib/api";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  BarChart, Bar
} from "recharts";

// ── Types ────────────────────────────────────────────────────────────────────
interface Health { status: string; backend: string; python: string }

// ── Static model data (architecture, not live inference results) ─────────────
const MODELS = [
  { name: "OSNet", type: "CNN", params: "2.2M", status: "ready", color: "var(--accent-blue)" },
  { name: "TransReID", type: "Transformer", params: "86M", status: "ready", color: "var(--accent-purple)" },
  { name: "CLIP ViT-B/16", type: "VLP", params: "149M", status: "ready", color: "var(--accent-cyan)" },
];

// ── Metric Card ──────────────────────────────────────────────────────────────
function MetricCard({
  label, value, sub, accent, loading
}: { label: string; value: string | number; sub: string; accent: string; loading?: boolean }) {
  return (
    <div className={`metric-card ${accent}`} style={{ animationDelay: "0ms" }}>
      <div className="metric-label">{label}</div>
      {loading
        ? <div className="skeleton" style={{ height: 36, width: "60%", margin: "8px 0 6px" }} />
        : <div className="metric-value" style={{ color: `var(--accent-${accent})` }}>{value}</div>
      }
      <div className="metric-sub">{sub}</div>
    </div>
  );
}

// ── Camera count bar chart data ───────────────────────────────────────────────
function buildCamData(stats: Record<string, DatasetStats>) {
  return Object.entries(stats).map(([name, s]) => ({
    name: name.replace("-", "\n"),
    cameras: s.cameras.length,
    identities: s.total_identities,
    images: Math.round((s.train_images + s.query_images + s.gallery_images) / 1000),
  }));
}

// ── Main ─────────────────────────────────────────────────────────────────────
export default function DashboardPage() {
  const [health, setHealth] = useState<Health | null>(null);
  const [datasetNames, setDatasetNames] = useState<string[]>([]);
  const [stats, setStats] = useState<Record<string, DatasetStats>>({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function load() {
      try {
        const [h, names] = await Promise.all([
          fetch("/api/health").then(r => r.json()),
          fetchDatasets(),
        ]);
        setHealth(h);
        setDatasetNames(names);

        const statsArr = await Promise.all(names.map(n => fetchDatasetStats(n)));
        const statsMap: Record<string, DatasetStats> = {};
        names.forEach((n, i) => (statsMap[n] = statsArr[i]));
        setStats(statsMap);
      } catch (e: unknown) {
        setError(e instanceof Error ? e.message : String(e));
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  const totalImages = Object.values(stats).reduce(
    (s, d) => s + d.train_images + d.query_images + d.gallery_images, 0
  );
  const totalIds = Object.values(stats).reduce((s, d) => s + d.total_identities, 0);
  const totalCams = Object.values(stats).reduce((s, d) => s + d.cameras.length, 0);
  const camData = buildCamData(stats);

  return (
    <>
      {/* Page Header */}
      <header className="page-header">
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700, lineHeight: 1 }}>
            <span className="gradient-text">M²-ReID</span> Research Platform
          </h1>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
            Adaptive Multi-Model Cross-Camera Person Re-Identification
          </p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {health && (
            <span className="badge badge-green">
              <span className="pulse-dot" style={{ width: 6, height: 6 }} />
              FastAPI · Python {health.python}
            </span>
          )}
          {error && <span className="badge" style={{ background: "rgba(239,68,68,0.15)", color: "#f87171", border: "1px solid rgba(239,68,68,0.3)" }}>⚠ {error}</span>}
        </div>
      </header>

      <main className="page-content animate-in">
        {/* Metric cards */}
        <div className="grid-4" style={{ marginBottom: 24 }}>
          <MetricCard
            label="Total Images"
            value={loading ? "—" : totalImages.toLocaleString()}
            sub="Across all datasets"
            accent="blue"
            loading={loading}
          />
          <MetricCard
            label="Total Identities"
            value={loading ? "—" : totalIds.toLocaleString()}
            sub="Unique person IDs"
            accent="cyan"
            loading={loading}
          />
          <MetricCard
            label="Dataset Cameras"
            value={loading ? "—" : totalCams}
            sub="Across all datasets"
            accent="purple"
            loading={loading}
          />
          <MetricCard
            label="Active Models"
            value={MODELS.length}
            sub="OSNet · TransReID · CLIP"
            accent="green"
          />
        </div>

        {/* Charts row */}
        <div className="grid-2" style={{ marginBottom: 24 }}>
          {/* Dataset breakdown */}
          <div className="glass-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 16 }}>
              <div className="section-title">Dataset Breakdown</div>
              <div className="section-subtitle">Images (×1k) per dataset</div>
            </div>
            {loading
              ? <div className="skeleton" style={{ height: 200 }} />
              : (
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={camData} barGap={6}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                    <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border-glow)", borderRadius: 8 }}
                      labelStyle={{ color: "var(--text-primary)" }}
                      itemStyle={{ color: "var(--text-secondary)" }}
                    />
                    <Bar dataKey="images" fill="var(--accent-blue)" radius={[4, 4, 0, 0]} name="Images (k)" />
                    <Bar dataKey="cameras" fill="var(--accent-cyan)" radius={[4, 4, 0, 0]} name="Cameras" />
                  </BarChart>
                </ResponsiveContainer>
              )}
          </div>

          {/* Identities area chart */}
          <div className="glass-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 16 }}>
              <div className="section-title">Identity Distribution</div>
              <div className="section-subtitle">Train / Query / Gallery split</div>
            </div>
            {loading
              ? <div className="skeleton" style={{ height: 200 }} />
              : (
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart
                    data={datasetNames.map(n => ({
                      name: n.replace("-", "\n"),
                      Train: stats[n]?.train_ids ?? 0,
                      Query: stats[n]?.query_ids ?? 0,
                      Gallery: stats[n]?.gallery_ids ?? 0,
                    }))}
                  >
                    <defs>
                      <linearGradient id="gTrain" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--accent-blue)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--accent-blue)" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="gQuery" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--accent-cyan)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--accent-cyan)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                    <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border-glow)", borderRadius: 8 }}
                      labelStyle={{ color: "var(--text-primary)" }}
                      itemStyle={{ color: "var(--text-secondary)" }}
                    />
                    <Area type="monotone" dataKey="Train" stroke="var(--accent-blue)" fill="url(#gTrain)" strokeWidth={2} />
                    <Area type="monotone" dataKey="Query" stroke="var(--accent-cyan)" fill="url(#gQuery)" strokeWidth={2} />
                  </AreaChart>
                </ResponsiveContainer>
              )}
          </div>
        </div>

        {/* Bottom row */}
        <div className="grid-2">
          {/* Dataset table */}
          <div className="glass-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 14 }}>
              <div className="section-title">Dataset Summary</div>
              <div className="section-subtitle">Live stats from dataset_validator.py</div>
            </div>
            {loading
              ? <div className="skeleton" style={{ height: 120 }} />
              : (
                <table className="data-table">
                  <thead>
                    <tr>
                      <th>Dataset</th>
                      <th>Images</th>
                      <th>IDs</th>
                      <th>Cams</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {datasetNames.map(n => {
                      const s = stats[n];
                      const total = s ? s.train_images + s.query_images + s.gallery_images : 0;
                      return (
                        <tr key={n}>
                          <td><span className="mono" style={{ color: "var(--accent-blue)", fontSize: 12 }}>{n}</span></td>
                          <td>{total.toLocaleString()}</td>
                          <td>{s?.total_identities.toLocaleString() ?? "—"}</td>
                          <td>{s?.cameras.length ?? "—"}</td>
                          <td>
                            <span className={`badge ${s?.valid ? "badge-green" : "badge-purple"}`}>
                              {s?.valid ? "✓ Valid" : "⚠ Error"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              )}
          </div>

          {/* Models panel */}
          <div className="glass-card" style={{ padding: 20 }}>
            <div style={{ marginBottom: 14 }}>
              <div className="section-title">Re-ID Model Stack</div>
              <div className="section-subtitle">Ensemble architecture</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {MODELS.map(m => (
                <div key={m.name} style={{
                  display: "flex", alignItems: "center", justifyContent: "space-between",
                  padding: "12px 16px",
                  background: "rgba(255,255,255,0.02)",
                  border: `1px solid ${m.color}30`,
                  borderRadius: 8,
                }}>
                  <div>
                    <div style={{ fontWeight: 600, fontSize: 13, color: m.color }}>{m.name}</div>
                    <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>{m.type} · {m.params} params</div>
                  </div>
                  <span className="badge badge-green">✓ Ready</span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
