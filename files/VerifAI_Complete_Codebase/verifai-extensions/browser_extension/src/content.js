(function () {
  if (window.__verifaiInjected) return;
  window.__verifaiInjected = true;

  let panel = null;

  function createPanel() {
    if (panel) panel.remove();

    panel = document.createElement("div");
    panel.id = "verifai-panel";
    panel.innerHTML = `
      <div id="vf-header">
        <span id="vf-logo">Verif<span style="color:#185FA5">AI</span></span>
        <button id="vf-close" aria-label="Close">✕</button>
      </div>
      <div id="vf-body">
        <div id="vf-status" class="vf-loading">
          <div class="vf-spinner"></div>
          <span id="vf-status-text">Analysing...</span>
        </div>
      </div>
    `;
    document.body.appendChild(panel);
    document.getElementById("vf-close").onclick = () => panel.remove();
    return panel;
  }

  const VERDICT_CFG = {
    fake:        { label: "Fake",           color: "#791F1F", bg: "#FCEBEB", bar: "#E24B4A" },
    likely_fake: { label: "Likely fake",    color: "#633806", bg: "#FAEEDA", bar: "#EF9F27" },
    uncertain:   { label: "Uncertain",      color: "#5F5E5A", bg: "#F1EFE8", bar: "#888780" },
    likely_real: { label: "Likely real",    color: "#27500A", bg: "#EAF3DE", bar: "#1D9E75" },
    real:        { label: "Authentic",      color: "#085041", bg: "#E1F5EE", bar: "#1D9E75" },
  };

  function showLoading(message) {
    createPanel();
    document.getElementById("vf-body").innerHTML = `
      <div class="vf-loading">
        <div class="vf-spinner"></div>
        <span>${message || "Analysing..."}</span>
      </div>
    `;
  }

  function showResult(result) {
    createPanel();
    const v = result.final_verdict || "uncertain";
    const cfg = VERDICT_CFG[v] || VERDICT_CFG.uncertain;
    const conf = Math.round((result.final_confidence || 0) * 100);
    const signals = (result.all_signals || []).slice(0, 3);

    document.getElementById("vf-body").innerHTML = `
      <div id="vf-verdict-row">
        <span class="vf-badge" style="background:${cfg.bg};color:${cfg.color}">${cfg.label}</span>
        <span class="vf-conf" style="color:${cfg.color}">${conf}%</span>
      </div>
      <p id="vf-reasoning">${(result.final_reasoning || "").slice(0, 220)}${result.final_reasoning?.length > 220 ? "..." : ""}</p>
      ${signals.length > 0 ? `
        <div class="vf-signals-label">Key signals</div>
        ${signals.map(s => `<div class="vf-signal"><span style="color:${cfg.bar}">›</span> ${s}</div>`).join("")}
      ` : ""}
      <div id="vf-type-row">${result.content_type || ""} · analysed just now</div>
      <a href="http://localhost:5173" target="_blank" id="vf-open-link">Open full report ↗</a>
    `;
  }

  function showError(message) {
    createPanel();
    document.getElementById("vf-body").innerHTML = `
      <div class="vf-error">
        <div class="vf-error-icon">!</div>
        <p>${message}</p>
        <p class="vf-error-hint">Make sure VerifAI backend is running:<br><code>uvicorn main:app --port 8000</code></p>
      </div>
    `;
  }

  chrome.runtime.onMessage.addListener((msg) => {
    if (msg.type === "VERIFAI_LOADING") showLoading(msg.message);
    if (msg.type === "VERIFAI_RESULT")  showResult(msg.result);
    if (msg.type === "VERIFAI_ERROR")   showError(msg.message);
  });

  // Keyboard shortcut: Alt+V to analyse current page
  document.addEventListener("keydown", (e) => {
    if (e.altKey && e.key === "v") {
      const selected = window.getSelection()?.toString().trim();
      chrome.runtime.sendMessage({
        type: "VERIFAI_ANALYSE_TEXT",
        text: selected || "",
        url: selected ? null : window.location.href,
        contentType: "news",
      });
    }
  });
})();
