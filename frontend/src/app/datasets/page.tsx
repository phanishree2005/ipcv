"use client";
import { useEffect, useState } from "react";
import { fetchDatasets, fetchDatasetStats, DatasetStats } from "@/lib/api";

export default function DatasetsPage() {
  const [names, setNames] = useState<string[]>([]);
  const [stats, setStats] = useState<Record<string, DatasetStats>>({});
  const [selected, setSelected] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      const ns = await fetchDatasets();
      setNames(ns);
      if (ns.length) setSelected(ns[0]);
      const arr = await Promise.all(ns.map(n => fetchDatasetStats(n)));
      const m: Record<string, DatasetStats> = {};
      ns.forEach((n, i) => (m[n] = arr[i]));
      setStats(m);
      setLoading(false);
    }
    load();
  }, []);

  const s = selected ? stats[selected] : null;

  return (
    <>
      <header className="page-header">
        <div>
          <h1 style={{ fontSize: 18, fontWeight: 700 }}>Datasets</h1>
          <p style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 3 }}>
            Live statistics from <span className="mono" style={{ color: "var(--accent-blue)", fontSize: 10 }}>src/data/dataset_validator.py</span>
          </p>
        </div>
      </header>

      <main className="page-content animate-in">
        {/* Dataset tabs */}
        <div style={{ display: "flex", gap: 8, marginBottom: 20 }}>
          {names.map(n => (
            <button
              key={n}
              onClick={() => setSelected(n)}
              style={{
                padding: "8px 20px", borderRadius: 8, border: "1px solid",
                borderColor: selected === n ? "var(--accent-blue)" : "var(--border)",
                background: selected === n ? "rgba(59,130,246,0.15)" : "transparent",
                color: selected === n ? "var(--accent-blue)" : "var(--text-secondary)",
                cursor: "pointer", fontSize: 13, fontWeight: 600,
                transition: "all 0.15s",
              }}
            >
              {n}
            </button>
          ))}
        </div>

        {loading || !s ? (
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            {[...Array(6)].map((_, i) => (
              <div key={i} className="skeleton" style={{ height: 80, borderRadius: 12 }} />
            ))}
          </div>
        ) : (
          <>
            {/* Status bar */}
            <div className="glass-card" style={{ padding: "14px 20px", marginBottom: 20, display: "flex", alignItems: "center", gap: 16 }}>
              <span className={`badge ${s.valid ? "badge-green" : "badge-purple"}`}>
                {s.valid ? "✓ Dataset Valid" : "⚠ Validation Errors"}
              </span>
              <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Path: <span className="mono" style={{ color: "var(--text-secondary)", fontSize: 11 }}>{s.root_path}</span>
              </span>
              {s.corrupted_images > 0 && (
                <span style={{ fontSize: 12, color: "var(--accent-orange)" }}>
                  ⚠ {s.corrupted_images} corrupted images
                </span>
              )}
            </div>

            {/* Stats grid */}
            <div className="grid-3" style={{ marginBottom: 20 }}>
              {[
                { label: "Train Images", value: s.train_images.toLocaleString(), sub: `${s.train_ids} identities`, accent: "blue" },
                { label: "Query Images", value: s.query_images.toLocaleString(), sub: `${s.query_ids} identities`, accent: "cyan" },
                { label: "Gallery Images", value: s.gallery_images.toLocaleString(), sub: `${s.gallery_ids} identities`, accent: "purple" },
                { label: "Total Identities", value: s.total_identities.toLocaleString(), sub: "Unique persons", accent: "green" },
                { label: "Cameras", value: s.cameras.length, sub: `IDs: ${s.cameras.join(", ")}`, accent: "blue" },
                { label: "Total Images", value: (s.train_images + s.query_images + s.gallery_images).toLocaleString(), sub: "All splits combined", accent: "cyan" },
              ].map(({ label, value, sub, accent }) => (
                <div key={label} className={`metric-card ${accent}`}>
                  <div className="metric-label">{label}</div>
                  <div className="metric-value" style={{ color: `var(--accent-${accent})`, fontSize: 24 }}>{value}</div>
                  <div className="metric-sub">{sub}</div>
                </div>
              ))}
            </div>

            {/* Camera IDs */}
            <div className="glass-card" style={{ padding: 20 }}>
              <div className="section-title" style={{ marginBottom: 12 }}>Camera IDs in {selected}</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 8 }}>
                {s.cameras.map(c => (
                  <div key={c} style={{
                    padding: "6px 14px", borderRadius: 8,
                    background: "rgba(6,182,212,0.1)",
                    border: "1px solid rgba(6,182,212,0.3)",
                    color: "var(--accent-cyan)",
                    fontSize: 13, fontWeight: 700,
                    fontFamily: "JetBrains Mono, monospace",
                  }}>
                    C{String(c).padStart(2, "0")}
                  </div>
                ))}
              </div>
            </div>

            {/* Errors */}
            {s.errors && s.errors.length > 0 && (
              <div className="glass-card" style={{ padding: 20, marginTop: 16, borderColor: "rgba(239,68,68,0.3)" }}>
                <div className="section-title" style={{ color: "#f87171", marginBottom: 10 }}>Validation Errors</div>
                {s.errors.map((e, i) => (
                  <div key={i} className="mono" style={{ fontSize: 11, color: "#f87171", padding: "4px 0", borderBottom: "1px solid rgba(239,68,68,0.1)" }}>
                    {e}
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </main>
    </>
  );
}
