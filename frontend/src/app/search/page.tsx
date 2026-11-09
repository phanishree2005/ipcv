"use client";

import { useEffect, useState, useRef } from "react";
import {
  fetchRetrievalSamples,
  searchRetrievalQuery,
  RetrievalSample,
  RetrievalResponse,
  RetrievalResultItem
} from "@/lib/api";

export default function ForensicSearchPage() {
  const [samples, setSamples] = useState<RetrievalSample[]>([]);
  const [selectedSample, setSelectedSample] = useState<RetrievalSample | null>(null);
  const [searchResult, setSearchResult] = useState<RetrievalResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [topK, setTopK] = useState(10);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    fetchRetrievalSamples()
      .then((data) => {
        setSamples(data);
        if (data.length > 0) {
          handleSelectSample(data[0]);
        }
      })
      .catch((err) => {
        console.error("Failed to load retrieval samples", err);
        setError("Could not load sample queries from backend. Ensure port 8001 is running.");
      });
  }, []);

  const handleSelectSample = async (s: RetrievalSample) => {
    setSelectedSample(s);
    setLoading(true);
    setError(null);
    try {
      const res = await searchRetrievalQuery({
        query_filename: s.filename,
        query_cam: s.cam,
        top_k: topK,
      });
      setSearchResult(res);
    } catch (e: any) {
      setError(e.message || "Retrieval search failed");
    } finally {
      setLoading(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = reader.result as string;
      setSelectedSample(null);
      setLoading(true);
      setError(null);
      try {
        const res = await searchRetrievalQuery({
          query_b64: b64,
          query_cam: 1,
          top_k: topK,
        });
        setSearchResult(res);
      } catch (err: any) {
        setError(err.message || "Custom query retrieval failed");
      } finally {
        setLoading(false);
      }
    };
    reader.readAsDataURL(file);
  };

  return (
    <div style={{ padding: "24px 32px", maxWidth: 1400, margin: "0 auto" }}>
      {/* Header */}
      <div style={{ marginBottom: 24 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 24 }}>🔍</span>
          <h1 style={{ fontSize: 22, fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
            Forensic Suspect Re-Identification Search
          </h1>
          <span className="badge badge-cyan" style={{ fontSize: 11 }}>M²-ReID XAI Engine</span>
        </div>
        <p style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>
          Cross-camera gallery retrieval conditioned on dynamic quality gating, anatomical pose keypoints, and camera transition priors.
        </p>
      </div>

      {error && (
        <div style={{ padding: "12px 16px", background: "rgba(239, 68, 68, 0.15)", border: "1px solid #ef4444", borderRadius: 8, color: "#fca5a5", fontSize: 12, marginBottom: 20 }}>
          ⚠️ {error}
        </div>
      )}

      {/* Sample Query Picker */}
      <div className="card" style={{ marginBottom: 24, padding: 16 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 12 }}>
          <span style={{ fontSize: 12, fontWeight: 600, color: "var(--text-primary)", letterSpacing: "0.05em", textTransform: "uppercase" }}>
            Select Target Suspect (Market-1501 Query Pool)
          </span>
          <div>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              accept="image/*"
              style={{ display: "none" }}
            />
            <button
              className="btn btn-secondary"
              onClick={() => fileInputRef.current?.click()}
              style={{ fontSize: 11, padding: "5px 12px" }}
            >
              📁 Upload Custom Suspect Crop
            </button>
          </div>
        </div>

        <div style={{ display: "flex", gap: 12, overflowX: "auto", paddingBottom: 6 }}>
          {samples.map((s) => {
            const isSelected = selectedSample?.filename === s.filename;
            return (
              <button
                key={s.filename}
                onClick={() => handleSelectSample(s)}
                style={{
                  background: isSelected ? "rgba(56, 189, 248, 0.18)" : "var(--bg-secondary)",
                  border: isSelected ? "2px solid var(--accent-cyan)" : "1px solid var(--border)",
                  borderRadius: 8,
                  padding: 6,
                  cursor: "pointer",
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  minWidth: 80,
                  transition: "all 0.15s ease",
                }}
              >
                <img
                  src={`data:image/jpeg;base64,${s.thumbnail_b64}`}
                  alt={`PID ${s.pid}`}
                  style={{ width: 56, height: 112, objectFit: "cover", borderRadius: 4 }}
                />
                <span style={{ fontSize: 10, fontWeight: 700, color: isSelected ? "var(--accent-cyan)" : (s.is_video_sighting ? "var(--accent-purple)" : "var(--text-primary)"), marginTop: 4, textAlign: "center" }}>
                  {s.label || `ID #${s.pid}`}
                </span>
                <span style={{ fontSize: 9, color: "var(--text-muted)" }}>Cam {s.cam}</span>
                {s.is_video_sighting && (
                  <span style={{ fontSize: 8, background: "#7c3aed", color: "#fff", padding: "1px 4px", borderRadius: 3, marginTop: 2 }}>
                    VIDEO
                  </span>
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Main Two-Column Layout */}
      <div style={{ display: "grid", gridTemplateColumns: "380px 1fr", gap: 24 }}>
        {/* Left Column: Query Analysis & Explainable AI */}
        <div>
          <div className="card" style={{ padding: 20 }}>
            <h2 style={{ fontSize: 13, fontWeight: 600, color: "var(--text-primary)", marginBottom: 14, textTransform: "uppercase", letterSpacing: "0.05em" }}>
              Target Query Analysis
            </h2>

            {searchResult ? (
              <>
                <div style={{ display: "flex", gap: 16, marginBottom: 18 }}>
                  <div style={{ position: "relative" }}>
                    <img
                      src={`data:image/jpeg;base64,${searchResult.query.image_b64}`}
                      alt="Query"
                      style={{ width: 110, height: 220, objectFit: "cover", borderRadius: 6, border: "2px solid var(--accent-cyan)" }}
                    />
                    <div style={{
                      position: "absolute", bottom: 6, left: 6, right: 6,
                      background: "rgba(0,0,0,0.8)", padding: "2px 6px", borderRadius: 4,
                      fontSize: 10, textAlign: "center", color: "#fff", fontWeight: 600
                    }}>
                      Cam {searchResult.query.camera}
                    </div>
                  </div>

                  <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 8 }}>
                    <div style={{ background: "var(--bg-secondary)", padding: "10px 12px", borderRadius: 6 }}>
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Target Identity</div>
                      <div style={{ fontSize: 18, fontWeight: 800, color: "var(--accent-cyan)" }}>
                        {searchResult.query.pid > 0 ? `Person #${searchResult.query.pid}` : "Custom Probe"}
                      </div>
                    </div>

                    <div style={{ background: "var(--bg-secondary)", padding: "10px 12px", borderRadius: 6 }}>
                      <div style={{ fontSize: 11, color: "var(--text-muted)" }}>Gallery Search Space</div>
                      <div style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)" }}>
                        {searchResult.total_gallery_searched} Candidates
                      </div>
                    </div>
                  </div>
                </div>

                {/* Adaptive Neural Gating Weights */}
                <div style={{ background: "var(--bg-secondary)", padding: 14, borderRadius: 8, marginBottom: 16 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", marginBottom: 10, display: "flex", justifyContent: "space-between" }}>
                    <span>Adaptive Gating Weights</span>
                    <span className="badge badge-purple" style={{ fontSize: 9 }}>Dynamic Gating</span>
                  </div>

                  <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                        <span style={{ color: "var(--accent-cyan)" }}>OSNet (CNN Spatial)</span>
                        <span style={{ fontWeight: 700 }}>{(searchResult.query.weights.osnet * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${searchResult.query.weights.osnet * 100}%`, background: "var(--accent-cyan)" }} />
                      </div>
                    </div>

                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                        <span style={{ color: "#c084fc" }}>TransReID (ViT Global)</span>
                        <span style={{ fontWeight: 700 }}>{(searchResult.query.weights.transreid * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${searchResult.query.weights.transreid * 100}%`, background: "#c084fc" }} />
                      </div>
                    </div>

                    <div>
                      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, marginBottom: 2 }}>
                        <span style={{ color: "#fbbf24" }}>CLIP (Zero-Shot VL)</span>
                        <span style={{ fontWeight: 700 }}>{(searchResult.query.weights.clip * 100).toFixed(1)}%</span>
                      </div>
                      <div style={{ height: 6, background: "rgba(255,255,255,0.08)", borderRadius: 3, overflow: "hidden" }}>
                        <div style={{ height: "100%", width: `${searchResult.query.weights.clip * 100}%`, background: "#fbbf24" }} />
                      </div>
                    </div>
                  </div>
                </div>

                {/* 7-D Quality Vector Breakdown */}
                <div style={{ background: "var(--bg-secondary)", padding: 14, borderRadius: 8 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", marginBottom: 8 }}>
                    Conditioning Image Quality Vector
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "6px 12px", fontSize: 10 }}>
                    <div style={{ color: "var(--text-secondary)" }}>Sharpness: <strong style={{ color: "var(--text-primary)" }}>{searchResult.query.quality.sharpness}</strong></div>
                    <div style={{ color: "var(--text-secondary)" }}>Contrast: <strong style={{ color: "var(--text-primary)" }}>{searchResult.query.quality.contrast}</strong></div>
                    <div style={{ color: "var(--text-secondary)" }}>Brightness: <strong style={{ color: "var(--text-primary)" }}>{searchResult.query.quality.brightness}</strong></div>
                    <div style={{ color: "var(--text-secondary)" }}>Confidence: <strong style={{ color: "var(--text-primary)" }}>{searchResult.query.quality.detection_conf}</strong></div>
                    <div style={{ color: "var(--text-secondary)" }}>Bbox Area: <strong style={{ color: "var(--text-primary)" }}>{searchResult.query.quality.bbox_area}</strong></div>
                    <div style={{ color: "var(--text-secondary)" }}>Occlusion: <strong style={{ color: searchResult.query.quality.occlusion_score > 0.4 ? "#f87171" : "#4ade80" }}>{searchResult.query.quality.occlusion_score}</strong></div>
                  </div>
                </div>
              </>
            ) : (
              <div style={{ textAlign: "center", padding: "40px 0", color: "var(--text-muted)", fontSize: 12 }}>
                {loading ? "Extracting multi-model features..." : "Select a query above"}
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Top-K Ranked Gallery Matches */}
        <div>
          <div className="card" style={{ padding: 20 }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
              <div>
                <h2 style={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", margin: 0 }}>
                  Top Ranked Gallery Matches (Cross-Camera Candidates)
                </h2>
                <div style={{ fontSize: 11, color: "var(--text-muted)", marginTop: 2 }}>
                  Sorted by composite score (M²-ReID Adaptive Embedding Cosine Similarity + Spatial-Temporal Prior)
                </div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Top-K:</span>
                {[5, 10, 15].map((k) => (
                  <button
                    key={k}
                    onClick={() => {
                      setTopK(k);
                      if (selectedSample) handleSelectSample(selectedSample);
                    }}
                    style={{
                      padding: "3px 8px",
                      borderRadius: 4,
                      fontSize: 10,
                      fontWeight: 600,
                      background: topK === k ? "var(--accent-cyan)" : "var(--bg-secondary)",
                      color: topK === k ? "#000" : "var(--text-secondary)",
                      border: "none",
                      cursor: "pointer"
                    }}
                  >
                    {k}
                  </button>
                ))}
              </div>
            </div>

            {loading ? (
              <div style={{ textAlign: "center", padding: "60px 0", color: "var(--accent-cyan)" }}>
                <div className="pulse-dot" style={{ margin: "0 auto 12px" }} />
                <span style={{ fontSize: 12 }}>Querying FAISS gallery with adaptive multi-model fusion...</span>
              </div>
            ) : searchResult && searchResult.results.length > 0 ? (
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(170px, 1fr))", gap: 14 }}>
                {searchResult.results.map((res: RetrievalResultItem) => {
                  const isRank1 = res.rank === 1;
                  const isCorrect = res.is_ground_truth_match;

                  return (
                    <div
                      key={res.filename}
                      style={{
                        background: isRank1 ? "rgba(56, 189, 248, 0.08)" : "var(--bg-secondary)",
                        border: isRank1
                          ? "2px solid var(--accent-cyan)"
                          : isCorrect
                          ? "1px solid rgba(74, 222, 128, 0.4)"
                          : "1px solid var(--border)",
                        borderRadius: 8,
                        padding: 10,
                        position: "relative",
                        display: "flex",
                        flexDirection: "column",
                        alignItems: "center"
                      }}
                    >
                      {/* Rank Tag */}
                      <div style={{
                        position: "absolute", top: 8, left: 8,
                        background: isRank1 ? "var(--accent-cyan)" : "rgba(0,0,0,0.7)",
                        color: isRank1 ? "#000" : "#fff",
                        fontWeight: 800, fontSize: 10, padding: "2px 6px", borderRadius: 4
                      }}>
                        #{res.rank}
                      </div>

                      {/* True Match Verification Tag */}
                      {res.is_ground_truth_match && (
                        <div style={{
                          position: "absolute", top: 8, right: 8,
                          background: "#22c55e", color: "#fff",
                          fontWeight: 700, fontSize: 9, padding: "2px 5px", borderRadius: 4
                        }}>
                          MATCH ✓
                        </div>
                      )}

                      <img
                        src={`data:image/jpeg;base64,${res.thumbnail_b64}`}
                        alt={res.filename}
                        style={{ width: 80, height: 160, objectFit: "cover", borderRadius: 4, marginTop: 14, marginBottom: 8 }}
                      />

                      <div style={{ width: "100%", fontSize: 11, textAlign: "center" }}>
                        <div style={{ fontWeight: 700, color: "var(--text-primary)" }}>
                          Person #{res.pid}
                        </div>
                        <div style={{ fontSize: 10, color: "var(--text-muted)", margin: "2px 0 6px" }}>
                          Camera {res.camera}
                        </div>

                        {/* Similarity Meter */}
                        <div style={{ background: "rgba(0,0,0,0.3)", padding: "4px 8px", borderRadius: 4 }}>
                          <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, fontWeight: 700 }}>
                            <span style={{ color: "var(--text-secondary)" }}>Similarity:</span>
                            <span style={{ color: res.similarity >= 80 ? "#4ade80" : "var(--accent-cyan)" }}>
                              {res.similarity}%
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : (
              <div style={{ textAlign: "center", padding: "60px 0", color: "var(--text-muted)", fontSize: 12 }}>
                No gallery candidates retrieved.
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
