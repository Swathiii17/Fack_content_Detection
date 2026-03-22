const API_BASE = "http://localhost:8000";

const VERDICT_CFG = {
  fake:        { label: "Fake",        color: "#791F1F", bg: "#FCEBEB" },
  likely_fake: { label: "Likely fake", color: "#633806", bg: "#FAEEDA" },
  uncertain:   { label: "Uncertain",   color: "#5F5E5A", bg: "#F1EFE8" },
  likely_real: { label: "Likely real", color: "#27500A", bg: "#EAF3DE" },
  real:        { label: "Authentic",   color: "#085041", bg: "#E1F5EE" },
};

let selectedType = "news";

// Check backend health
async function checkBackend() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
    const ok = res.ok;
    document.getElementById("status-dot").style.background = ok ? "#639922" : "#E24B4A";
    document.getElementById("status-text").textContent = ok ? "Backend online" : "Backend offline";
  } catch {
    document.getElementById("status-dot").style.background = "#E24B4A";
    document.getElementById("status-text").textContent = "Backend offline";
  }
}

// Type toggle
document.querySelectorAll(".type-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".type-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    selectedType = btn.dataset.type;
  });
});

// Analyse page
document.getElementById("btn-page").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.runtime.sendMessage({
    type: "VERIFAI_ANALYSE_TEXT",
    text: "",
    url: tab.url,
    contentType: "news",
  });
  window.close();
});

// Analyse selection
document.getElementById("btn-selection").addEventListener("click", async () => {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const [result] = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection()?.toString().trim() || "",
  });
  const selected = result?.result || "";
  if (!selected) {
    alert("Please select some text on the page first.");
    return;
  }
  chrome.runtime.sendMessage({
    type: "VERIFAI_ANALYSE_TEXT",
    text: selected,
    contentType: "news",
  });
  window.close();
});

// Analyse pasted
document.getElementById("btn-paste").addEventListener("click", async () => {
  const text = document.getElementById("paste-input").value.trim();
  if (!text) { alert("Please paste some content first."); return; }
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  chrome.runtime.sendMessage({
    type: "VERIFAI_ANALYSE_TEXT",
    text,
    contentType: selectedType,
  });
  window.close();
});

// Load and show last result
async function loadLastResult() {
  const { lastResult, lastResultTime } = await chrome.storage.local.get(["lastResult", "lastResultTime"]);
  if (!lastResult) return;

  const age = lastResultTime ? Math.floor((Date.now() - lastResultTime) / 60000) : null;
  const v = lastResult.final_verdict || "uncertain";
  const cfg = VERDICT_CFG[v] || VERDICT_CFG.uncertain;
  const conf = Math.round((lastResult.final_confidence || 0) * 100);

  const section = document.getElementById("last-result-section");
  section.style.display = "block";
  section.innerHTML = `
    <div class="last-result" style="border-top:0.5px solid rgba(0,0,0,0.06);">
      <div class="section-label" style="margin-bottom:6px;">Last result${age !== null ? ` · ${age < 1 ? "just now" : age + "m ago"}` : ""}</div>
      <span class="result-badge" style="background:${cfg.bg};color:${cfg.color}">${cfg.label} · ${conf}%</span>
      <div class="result-reasoning">${(lastResult.final_reasoning || "").slice(0, 160)}...</div>
      <div class="result-meta">${lastResult.content_type || ""}</div>
    </div>
  `;
}

checkBackend();
loadLastResult();
