const API_BASE = "http://localhost:8000/api";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "verifai-page",
    title: "Check this page with VerifAI",
    contexts: ["page"],
  });
  chrome.contextMenus.create({
    id: "verifai-selection",
    title: "Check selected text with VerifAI",
    contexts: ["selection"],
  });
  chrome.contextMenus.create({
    id: "verifai-image",
    title: "Check this image with VerifAI",
    contexts: ["image"],
  });
  chrome.contextMenus.create({
    id: "verifai-link",
    title: "Check this link with VerifAI",
    contexts: ["link"],
  });
});

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === "verifai-selection") {
    await analyseText(info.selectionText, "news", null, tab.id);
  } else if (info.menuItemId === "verifai-page") {
    await analyseText("", "news", tab.url, tab.id);
  } else if (info.menuItemId === "verifai-link") {
    await analyseText("", "news", info.linkUrl, tab.id);
  } else if (info.menuItemId === "verifai-image") {
    await analyseImageUrl(info.srcUrl, tab.id);
  }
});

async function analyseText(text, contentType, url, tabId) {
  await sendToContent(tabId, { type: "VERIFAI_LOADING", message: "Analysing content..." });

  try {
    const form = new FormData();
    form.append("text", text || "");
    form.append("content_type", contentType);
    if (url) form.append("url", url);

    const res = await fetch(`${API_BASE}/analyse/text`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const result = await res.json();

    await chrome.storage.local.set({
      lastResult: result,
      lastResultTime: Date.now(),
    });
    await sendToContent(tabId, { type: "VERIFAI_RESULT", result });
  } catch (err) {
    await sendToContent(tabId, {
      type: "VERIFAI_ERROR",
      message: "Could not reach VerifAI. Is the backend running at localhost:8000?",
    });
  }
}

async function analyseImageUrl(imageUrl, tabId) {
  await sendToContent(tabId, { type: "VERIFAI_LOADING", message: "Fetching and analysing image..." });

  try {
    const imgRes = await fetch(imageUrl);
    const blob = await imgRes.blob();
    const filename = imageUrl.split("/").pop()?.split("?")[0] || "image.jpg";

    const form = new FormData();
    form.append("file", blob, filename);

    const res = await fetch(`${API_BASE}/analyse/image`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    const result = await res.json();

    await chrome.storage.local.set({
      lastResult: result,
      lastResultTime: Date.now(),
    });
    await sendToContent(tabId, { type: "VERIFAI_RESULT", result });
  } catch (err) {
    await sendToContent(tabId, {
      type: "VERIFAI_ERROR",
      message: "Image analysis failed. Check CORS settings or try a direct image URL.",
    });
  }
}

async function sendToContent(tabId, message) {
  try {
    await chrome.tabs.sendMessage(tabId, message);
  } catch {
    // Content script may not be ready yet — store for popup
    await chrome.storage.session.set({ pendingMessage: message });
  }
}

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "VERIFAI_ANALYSE_TEXT") {
    analyseText(msg.text, msg.contentType || "news", msg.url || null, sender.tab?.id);
    sendResponse({ ok: true });
  }
  return true;
});
