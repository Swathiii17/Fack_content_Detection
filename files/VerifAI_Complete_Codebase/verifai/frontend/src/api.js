import axios from "axios";

const api = axios.create({ baseURL: "http://localhost:8000/api" });

export async function analyseText(text, contentType = "news", url = null) {
  const form = new FormData();
  form.append("text", text);
  form.append("content_type", contentType);
  if (url) form.append("url", url);
  const { data } = await api.post("/analyse/text", form);
  return data;
}

export async function analyseImage(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/analyse/image", form);
  return data;
}

export async function analyseAudio(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/analyse/audio", form);
  return data;
}

export async function getAnalyses() {
  const { data } = await api.get("/analyses");
  return data;
}

export async function getAnalysis(id) {
  const { data } = await api.get(`/analyses/${id}`);
  return data;
}
