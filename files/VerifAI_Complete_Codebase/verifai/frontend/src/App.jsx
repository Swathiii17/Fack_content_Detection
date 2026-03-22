import { useState, useEffect, useRef } from "react";
import { analyseText, analyseImage, analyseAudio, getAnalyses } from "./api";
import "./App.css";

const VERDICT_CONFIG = {
  fake:        { label: "Fake",           color: "#791F1F", bg: "#FCEBEB", bar: "#E24B4A" },
  likely_fake: { label: "Likely fake",    color: "#633806", bg: "#FAEEDA", bar: "#EF9F27" },
  uncertain:   { label: "Uncertain",      color: "#444441", bg: "#F1EFE8", bar: "#888780" },
  likely_real: { label: "Likely real",    color: "#27500A", bg: "#EAF3DE", bar: "#639922" },
  real:        { label: "Authentic",      color: "#085041", bg: "#E1F5EE", bar: "#1D9E75" },
};

function VerdictBadge({ verdict }) {
  const cfg = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.uncertain;
  return (
    <span style={{ background: cfg.bg, color: cfg.color, fontSize: 11, fontWeight: 500,
      padding: "3px 10px", borderRadius: 99 }}>
      {cfg.label}
    </span>
  );
}

function ScoreBar({ label, value, verdict }) {
  const cfg = VERDICT_CONFIG[verdict] || VERDICT_CONFIG.uncertain;
  const pct = Math.round((value ?? 0) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, margin: "5px 0" }}>
      <span style={{ fontSize: 12, color: "var(--text-2)", width: 150, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 5, background: "var(--bg-2)", borderRadius: 3, overflow: "hidden" }}>
        <div style={{ width: `${pct}%`, height: "100%", background: cfg.bar, borderRadius: 3 }} />
      </div>
      <span style={{ fontSize: 12, fontWeight: 500, minWidth: 34, textAlign: "right", color: cfg.color }}>
        {pct}%
      </span>
    </div>
  );
}

function ResultCard({ result, expanded, onToggle }) {
  if (!result) return null;
  const cfg = VERDICT_CONFIG[result.final_verdict] || VERDICT_CONFIG.uncertain;
  const agentResults = result.agent_results || {};

  return (
    <div className="card result-card" style={{ borderLeft: `3px solid ${cfg.bar}` }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
        <div>
          <div style={{ fontSize: 12, color: "var(--text-2)", marginBottom: 4 }}>
            {result.content_type} · {new Date(result.created_at).toLocaleTimeString()}
          </div>
          <VerdictBadge verdict={result.final_verdict} />
        </div>
        <span style={{ fontSize: 18, fontWeight: 500, color: cfg.color }}>
          {Math.round((result.final_confidence ?? 0) * 100)}%
        </span>
      </div>

      <p style={{ fontSize: 13, color: "var(--text-1)", lineHeight: 1.6, margin: "8px 0" }}>
        {result.final_reasoning}
      </p>

      {result.all_signals?.length > 0 && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11, fontWeight: 500, color: "var(--text-2)", marginBottom: 6 }}>
            Key signals
          </div>
          {result.all_signals.slice(0, 4).map((s, i) => (
            <div key={i} style={{ fontSize: 12, color: "var(--text-1)", padding: "3px 0",
              borderBottom: "0.5px solid var(--border)", display: "flex", gap: 6 }}>
              <span style={{ color: cfg.bar }}>›</span> {s}
            </div>
          ))}
        </div>
      )}

      {Object.keys(agentResults).length > 0 && (
        <div style={{ marginTop: 12 }}>
          <button className="btn-link" onClick={onToggle} style={{ fontSize: 12 }}>
            {expanded ? "Hide" : "Show"} agent details
          </button>
          {expanded && Object.entries(agentResults).map(([name, r]) => (
            <div key={name} style={{ marginTop: 12, paddingTop: 12, borderTop: "0.5px solid var(--border)" }}>
              <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 8 }}>
                <span style={{ fontSize: 13, fontWeight: 500 }}>{name} agent</span>
                <VerdictBadge verdict={r.verdict} />
              </div>
              {r.source_score !== undefined && <ScoreBar label="Source credibility" value={r.source_score} verdict={r.verdict} />}
              {r.claim_score !== undefined && <ScoreBar label="Claim verification" value={r.claim_score} verdict={r.verdict} />}
              {r.language_score !== undefined && <ScoreBar label="Language analysis" value={r.language_score} verdict={r.verdict} />}
              {r.specificity_score !== undefined && <ScoreBar label="Review specificity" value={r.specificity_score} verdict={r.verdict} />}
              {r.ai_generated_probability !== undefined && <ScoreBar label="AI generated" value={r.ai_generated_probability} verdict={r.verdict} />}
              {r.deepfake_probability !== undefined && <ScoreBar label="Deepfake probability" value={r.deepfake_probability} verdict={r.verdict} />}
              {r.visual_consistency_score !== undefined && <ScoreBar label="Visual consistency" value={r.visual_consistency_score} verdict={r.verdict} />}
              {r.ela_score !== undefined && <ScoreBar label="ELA manipulation score" value={r.ela_score} verdict={r.verdict} />}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState("text");
  const [contentType, setContentType] = useState("news");
  const [text, setText] = useState("");
  const [url, setUrl] = useState("");
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);
  const [expanded, setExpanded] = useState({});
  const [history, setHistory] = useState([]);
  const fileRef = useRef();

  useEffect(() => {
    getAnalyses().then(setHistory).catch(() => {});
  }, []);

  async function handleSubmit() {
    if (loading) return;
    setLoading(true);
    try {
      let result;
      if (tab === "text") {
        result = await analyseText(text, contentType, url || null);
      } else if (tab === "image") {
        result = await analyseImage(file);
      } else {
        result = await analyseAudio(file);
      }
      setResults(prev => [result, ...prev]);
      setHistory(prev => [result, ...prev]);
      setText(""); setUrl(""); setFile(null);
      if (fileRef.current) fileRef.current.value = "";
    } catch (e) {
      alert(e.response?.data?.detail || "Analysis failed. Check your API keys and backend.");
    } finally {
      setLoading(false);
    }
  }

  const canSubmit = tab === "text" ? (text.trim() || url.trim()) : !!file;
  const fakeCount = history.filter(h => ["fake","likely_fake"].includes(h.final_verdict)).length;
  const avgConf = history.length > 0
    ? Math.round(history.reduce((s, h) => s + (h.final_confidence || 0), 0) / history.length * 100)
    : 0;

  return (
    <div className="app">
      <header className="topbar">
        <div className="logo">Verif<span>AI</span></div>
        <div style={{ fontSize: 12, color: "var(--text-2)" }}>Journalist workspace</div>
      </header>

      <div className="stats-row">
        <div className="stat-card">
          <div className="stat-label">Analyses</div>
          <div className="stat-val">{history.length}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Flagged fake</div>
          <div className="stat-val" style={{ color: "#A32D2D" }}>{fakeCount}</div>
        </div>
        <div className="stat-card">
          <div className="stat-label">Avg confidence</div>
          <div className="stat-val">{avgConf}%</div>
        </div>
      </div>

      <div className="card">
        <div className="tab-row">
          {["text", "image", "audio"].map(t => (
            <button key={t} className={`tab ${tab === t ? "active" : ""}`} onClick={() => setTab(t)}>
              {t.charAt(0).toUpperCase() + t.slice(1)}
            </button>
          ))}
        </div>

        {tab === "text" && (
          <div>
            <div className="type-row">
              {["news", "review"].map(ct => (
                <button key={ct} className={`chip ${contentType === ct ? "active" : ""}`}
                  onClick={() => setContentType(ct)}>
                  {ct.charAt(0).toUpperCase() + ct.slice(1)}
                </button>
              ))}
            </div>
            <input className="input" placeholder="Paste a URL to analyse..."
              value={url} onChange={e => setUrl(e.target.value)} />
            <textarea className="textarea" placeholder={
              contentType === "news"
                ? "Or paste a news article, headline, or claim..."
                : "Or paste a review to analyse..."
            } value={text} onChange={e => setText(e.target.value)} rows={5} />
          </div>
        )}

        {(tab === "image" || tab === "audio") && (
          <div className="upload-zone" onClick={() => fileRef.current?.click()}>
            <div style={{ fontSize: 13, fontWeight: 500 }}>
              {file ? file.name : `Drop ${tab} file here or click to browse`}
            </div>
            <div style={{ fontSize: 12, color: "var(--text-2)", marginTop: 4 }}>
              {tab === "image" ? "JPEG, PNG, WebP" : "MP3, WAV, OGG, FLAC"}
            </div>
            <input ref={fileRef} type="file" style={{ display: "none" }}
              accept={tab === "image" ? "image/*" : "audio/*"}
              onChange={e => setFile(e.target.files[0])} />
          </div>
        )}

        <button className="btn-primary" onClick={handleSubmit}
          disabled={!canSubmit || loading} style={{ marginTop: 12, width: "100%" }}>
          {loading ? "Analysing..." : "Analyse content"}
        </button>
      </div>

      {results.length > 0 && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 500, margin: "1.5rem 0 0.75rem" }}>
            Latest results
          </div>
          {results.map((r, i) => (
            <ResultCard key={r.id || i} result={r}
              expanded={!!expanded[r.id]}
              onToggle={() => setExpanded(prev => ({ ...prev, [r.id]: !prev[r.id] }))} />
          ))}
        </div>
      )}

      {history.length > 0 && results.length === 0 && (
        <div>
          <div style={{ fontSize: 13, fontWeight: 500, margin: "1.5rem 0 0.75rem" }}>
            Recent analyses
          </div>
          {history.slice(0, 5).map((r, i) => (
            <div key={r.id || i} className="card" style={{ display: "flex",
              justifyContent: "space-between", alignItems: "center", padding: "0.75rem 1rem" }}>
              <div>
                <div style={{ fontSize: 12, color: "var(--text-2)" }}>{r.content_type} · {r.input_summary}</div>
              </div>
              <VerdictBadge verdict={r.final_verdict} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
