// Small wrapper around browser storage, so the app works even if storage is blocked.
const mem = {};
function read(area, key) {
  try { return window[area].getItem(key); } catch { return mem[area + key] ?? null; }
}
function write(area, key, value) {
  try {
    if (value === null || value === undefined) window[area].removeItem(key);
    else window[area].setItem(key, value);
  } catch {
    mem[area + key] = value ?? null;
  }
}
function readJson(area, key, fallback) {
  try { return JSON.parse(read(area, key)) ?? fallback; } catch { return fallback; }
}

const listeners = new Set();

export const store = {
  // Students stay signed in on their own device (localStorage).
  student() { return readJson("localStorage", "am.student", null); },
  studentToken() { return this.student()?.token || null; },
  setStudent(value) { write("localStorage", "am.student", value ? JSON.stringify(value) : null); listeners.forEach((fn) => fn()); },

  // Teachers are signed in only until the browser tab is closed (safer on shared classroom computers).
  teacherToken() { return read("sessionStorage", "am.teacher"); },
  setTeacherToken(token) { write("sessionStorage", "am.teacher", token); listeners.forEach((fn) => fn()); },

  prefs() { return readJson("localStorage", "am.prefs", {}); },
  setPrefs(patch) { write("localStorage", "am.prefs", JSON.stringify({ ...this.prefs(), ...patch })); },

  draft() { return readJson("localStorage", "am.draft", {}); },
  setDraft(patch) { write("localStorage", "am.draft", JSON.stringify({ ...this.draft(), ...patch })); },
  clearDraft() { write("localStorage", "am.draft", null); },

  theme() { return read("localStorage", "am.theme"); },
  setTheme(theme) { write("localStorage", "am.theme", theme); },

  onChange(fn) { listeners.add(fn); return () => listeners.delete(fn); },
};
