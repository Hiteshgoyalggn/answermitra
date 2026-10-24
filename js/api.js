// Talks to the AnswerMitra server (the Hugging Face Space). All requests go through here.
import { store } from "./store.js";

const settings = window.ANSWERMITRA_CONFIG || {};
export const API_BASE = String(settings.apiBase || "").replace(/\/+$/, "") + "/api/v1";

export class ApiError extends Error {
  constructor(message, status = 0) {
    super(message);
    this.status = status;
  }
}

const OFFLINE_MESSAGE =
  "Can't reach AnswerMitra right now. Check your internet connection. If the server was asleep, it may need a minute to wake up.";

function authHeader(auth) {
  const token = auth === "teacher" ? store.teacherToken() : auth === "student" ? store.studentToken() : null;
  return token ? { Authorization: `Bearer ${token}` } : {};
}

/**
 * Call the API. Options: method, body (JSON), auth ("student" | "teacher" | "optional"), timeout (ms).
 * Returns the parsed JSON, or throws ApiError with a message that can be shown to the user.
 */
export async function api(path, { method = "GET", body, auth, timeout = 200000 } = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  const headers = { ...authHeader(auth === "optional" ? (store.studentToken() ? "student" : null) : auth) };
  if (body !== undefined) headers["Content-Type"] = "application/json";
  let response;
  try {
    response = await fetch(API_BASE + path, {
      method, headers, body: body !== undefined ? JSON.stringify(body) : undefined, signal: controller.signal,
    });
  } catch (err) {
    throw new ApiError(err.name === "AbortError"
      ? "This is taking too long. Please try again in a minute." : OFFLINE_MESSAGE, 0);
  } finally {
    clearTimeout(timer);
  }
  let data = null;
  try { data = await response.json(); } catch { /* not JSON */ }
  if (!response.ok) {
    if (response.status === 401) {
      if (auth === "teacher") store.setTeacherToken(null);
      if (auth === "student" || auth === "optional") store.setStudent(null);
    }
    const fallback = response.status === 503 || response.status === 502
      ? "The server is starting up or busy. Please wait a moment and try again."
      : `Something went wrong (error ${response.status}). Please try again.`;
    throw new ApiError((data && data.error) || fallback, response.status);
  }
  return data;
}

/** Upload files with a progress callback (fetch can't report upload progress). */
export function uploadFiles(path, files, extraFields = {}, onProgress = () => {}) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f, f.name));
    Object.entries(extraFields).forEach(([k, v]) => v && form.append(k, v));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API_BASE + path);
    const token = store.teacherToken();
    if (token) xhr.setRequestHeader("Authorization", `Bearer ${token}`);
    xhr.timeout = 10 * 60 * 1000;
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress(e.loaded / e.total);
    xhr.onload = () => {
      let data = null;
      try { data = JSON.parse(xhr.responseText); } catch { /* ignore */ }
      if (xhr.status >= 200 && xhr.status < 300) resolve(data);
      else {
        if (xhr.status === 401) store.setTeacherToken(null);
        reject(new ApiError((data && data.error) || `Upload failed (error ${xhr.status}).`, xhr.status));
      }
    };
    xhr.onerror = () => reject(new ApiError(OFFLINE_MESSAGE, 0));
    xhr.ontimeout = () => reject(new ApiError("The upload took too long. Please try a smaller file.", 0));
    xhr.send(form);
  });
}

/** Download a file (like the CSV) that needs the teacher's sign-in. */
export async function downloadFile(path, filename) {
  let response;
  try {
    response = await fetch(API_BASE + path, { headers: authHeader("teacher") });
  } catch {
    throw new ApiError(OFFLINE_MESSAGE, 0);
  }
  if (!response.ok) {
    let data = null;
    try { data = await response.json(); } catch { /* ignore */ }
    throw new ApiError((data && data.error) || "The download failed. Please try again.", response.status);
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
