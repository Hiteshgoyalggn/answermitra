// Small UI helpers: safe element building, toasts, modals, formatting.

/**
 * Create an element safely. Text children are added as text (never as HTML),
 * so anything typed by students or returned by the AI can't inject code.
 *   h("button", { class: "btn", onclick: fn }, "Save")
 */
export function h(tag, attrs = {}, ...children) {
  const el = document.createElementNS(
    ["svg", "path", "circle", "line", "polyline", "polygon", "rect", "g", "text", "defs", "linearGradient", "stop"].includes(tag)
      ? "http://www.w3.org/2000/svg" : "http://www.w3.org/1999/xhtml", tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === false || value === null || value === undefined) continue;
    if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "class") el.setAttribute("class", value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (!(el instanceof SVGElement) && (key === "value" || (key in el && typeof value !== "string"))) el[key] = value;
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

/** Inline SVG icon from a few path strings. */
export function icon(paths, cls = "") {
  return h("svg", { viewBox: "0 0 24 24", "aria-hidden": "true", class: cls },
    ...paths.map((d) => h("path", { d })));
}
export const ICONS = {
  check: ["M5 12.5 10 17.5 19 7"],
  camera: ["M4 8h3l2-3h6l2 3h3v11H4z", "M12 17a4 4 0 1 0 0-8 4 4 0 0 0 0 8z"],
  pen: ["M4 20h4L19 9l-4-4L4 16z", "M13.5 6.5l4 4"],
  sparkle: ["M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5 18 18M18 6l-2.5 2.5M8.5 15.5 6 18"],
  refresh: ["M20 11a8 8 0 1 0-2.3 5.7", "M20 5v6h-6"],
  download: ["M12 4v11", "M7 10l5 5 5-5", "M5 20h14"],
  upload: ["M12 20V9", "M7 14l5-5 5 5", "M5 4h14"],
  logout: ["M15 4h4v16h-4", "M10 8l-4 4 4 4", "M6 12h10"],
  flag: ["M5 21V4", "M5 4h11l-2 4 2 4H5"],
  copy: ["M9 9h10v10H9z", "M5 15V5h10"],
  book: ["M4 19.5V6a2 2 0 0 1 2-2h12v15.5H6a2 2 0 0 0-2 2", "M8 8h7"],
  x: ["M6 6l12 12", "M18 6 6 18"],
  arrowRight: ["M5 12h14", "M13 6l6 6-6 6"],
};

/** Clear an element and add new children. */
export function mount(target, ...children) {
  target.replaceChildren(...children.flat(Infinity).filter(Boolean));
}

/** Toast notification. type: "info" | "success" | "error" | "warning". */
export function toast(message, type = "info", { timeout } = {}) {
  const region = document.getElementById("toasts");
  const icons = { info: "ℹ️", success: "✅", error: "⚠️", warning: "⏳" };
  const el = h("div", { class: `toast toast--${type}`, role: type === "error" ? "alert" : "status" },
    h("span", { class: "toast__icon", "aria-hidden": "true" }, icons[type] || "ℹ️"),
    h("div", { class: "toast__body" }, message),
    h("button", { class: "toast__close", type: "button", "aria-label": "Close message", onclick: () => close() }, "×"));
  region.append(el);
  const close = () => {
    el.classList.add("is-leaving");
    setTimeout(() => el.remove(), 200);
  };
  setTimeout(close, timeout ?? (type === "error" ? 8000 : 4500));
  return close;
}

/** Put a button into a busy state while a promise runs. Returns the promise result. */
export async function withBusy(button, label, task) {
  const original = [...button.childNodes];
  button.disabled = true;
  button.setAttribute("aria-busy", "true");
  button.replaceChildren(h("span", { class: "spinner", "aria-hidden": "true" }), label);
  try {
    return await task();
  } finally {
    button.disabled = false;
    button.removeAttribute("aria-busy");
    button.replaceChildren(...original);
  }
}

/** Open a modal dialog. Returns a close function. */
export function openModal(title, body, { onClose } = {}) {
  const root = document.getElementById("modal-root");
  const previousFocus = document.activeElement;
  const close = () => {
    backdrop.remove();
    document.body.classList.remove("has-modal");
    document.removeEventListener("keydown", onKey);
    previousFocus?.focus?.();
    onClose?.();
  };
  const onKey = (e) => { if (e.key === "Escape") close(); };
  const closeBtn = h("button", { class: "icon-btn", type: "button", "aria-label": "Close", onclick: close }, icon(ICONS.x));
  const dialog = h("div", { class: "modal", role: "dialog", "aria-modal": "true", "aria-label": title },
    h("div", { class: "modal__head" }, h("h2", {}, title), closeBtn),
    h("div", { class: "modal__body" }, body));
  const backdrop = h("div", { class: "modal-backdrop", onclick: (e) => e.target === backdrop && close() }, dialog);
  root.append(backdrop);
  document.body.classList.add("has-modal");
  document.addEventListener("keydown", onKey);
  closeBtn.focus();
  return close;
}

export function scoreClass(percent) {
  if (percent === null || percent === undefined) return "score--none";
  return percent >= 75 ? "score--high" : percent >= 50 ? "score--mid" : "score--low";
}

export function scorePill(percent) {
  return h("span", { class: `score-pill ${scoreClass(percent)}` },
    percent === null || percent === undefined ? "–" : `${percent}%`);
}

/** "2026-10-24 11:01:40 UTC" -> local, friendly date. */
export function formatDate(text, withTime = true) {
  if (!text) return "";
  const date = new Date(text.replace(" UTC", "Z").replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return text;
  return date.toLocaleString(undefined, withTime
    ? { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" }
    : { day: "numeric", month: "short", year: "numeric" });
}

export function kindLabel(kind) {
  return kind === "hots" ? "HOTS" : "Direct";
}

/** Character counter under a textarea. */
export function counter(input, max) {
  const el = h("span", { class: "counter", "aria-live": "polite" });
  const update = () => {
    const n = input.value.length;
    el.textContent = `${n} / ${max}`;
    el.classList.toggle("counter--warn", n > max * 0.95);
  };
  input.addEventListener("input", update);
  update();
  return el;
}

/** Radio group styled as a segmented control. */
export function segmented(name, options, value, onChange) {
  return h("div", { class: "segmented", role: "radiogroup" },
    options.map((opt) => h("label", {},
      h("input", { type: "radio", name, value: opt.value, checked: opt.value === value,
        onchange: (e) => e.target.checked && onChange(opt.value) }),
      h("span", {}, opt.label, opt.hint ? h("small", {}, opt.hint) : null))));
}

export function emptyState(title, text, action) {
  return h("div", { class: "empty" },
    h("svg", { viewBox: "0 0 64 64", "aria-hidden": "true" },
      h("rect", { x: "10", y: "12", width: "44", height: "40", rx: "8", fill: "var(--primary-soft)" }),
      h("path", { d: "M20 26h24M20 34h16M20 42h20", stroke: "var(--primary)", "stroke-width": "3", "stroke-linecap": "round" })),
    h("strong", {}, title), text ? h("p", { class: "muted small" }, text) : null, action || null);
}
