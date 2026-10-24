// AnswerMitra website: starts the app, wakes the server, and switches between pages.
import { api, API_BASE } from "./api.js";
import { store } from "./store.js";
import { h, mount, toast } from "./ui.js";
import { renderPractice } from "./views/practice.js";
import { renderProgress } from "./views/progress.js";
import { renderTeacher } from "./views/teacher.js";

const main = document.getElementById("main");
const banner = document.getElementById("server-banner");
const routes = { practice: renderPractice, progress: renderProgress, teacher: renderTeacher };
const titles = { practice: "Practice", progress: "My Progress", teacher: "Teacher" };
const ctx = { config: null, rerender: () => render(), reloadConfig };

// ---------- Theme ----------
document.getElementById("theme-toggle").addEventListener("click", () => {
  const current = document.documentElement.dataset.theme
    || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  store.setTheme(next);
  document.querySelector('meta[name="theme-color"]').content = next === "dark" ? "#17182b" : "#5b4bdb";
});

// ---------- Signed-in name in the top bar ----------
function updateUserChip() {
  const chip = document.getElementById("user-chip");
  const student = store.student();
  chip.hidden = !student;
  chip.textContent = student ? `👤 ${student.name}` : "";
}
store.onChange(updateUserChip);

// ---------- Server wake-up ----------
function showBanner(text, isError = false, action = null) {
  banner.hidden = false;
  banner.classList.toggle("server-banner--error", isError);
  mount(banner, isError ? null : h("span", { class: "spinner", "aria-hidden": "true" }), h("span", {}, text), action);
}

async function connect() {
  const slowTimer = setTimeout(() => showBanner(
    "Waking up AnswerMitra… Free servers sleep when nobody uses them, so this can take up to a minute."), 2500);
  const started = Date.now();
  while (Date.now() - started < 150000) {
    try {
      await api("/health", { timeout: 20000 });
      clearTimeout(slowTimer);
      banner.hidden = true;
      return true;
    } catch {
      await new Promise((r) => setTimeout(r, 4000));
    }
  }
  clearTimeout(slowTimer);
  showBanner("Can't reach the AnswerMitra server. Please check your internet connection, or ask your teacher.", true,
    h("button", { class: "btn btn--secondary btn--sm", type: "button", onclick: () => start() }, "Try again"));
  return false;
}

async function reloadConfig() {
  ctx.config = await api("/config");
  return ctx.config;
}

// ---------- Router ----------
function currentRoute() {
  const name = (location.hash.replace(/^#\/?/, "") || "practice").split("?")[0];
  return routes[name] ? name : "practice";
}

function render() {
  if (!ctx.config) return;
  const route = currentRoute();
  document.querySelectorAll("[data-route]").forEach((a) =>
    a.dataset.route === route ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"));
  document.title = `${titles[route]} · AnswerMitra`;
  main.replaceChildren();
  routes[route](main, ctx);
  main.focus({ preventScroll: true });
}

window.addEventListener("hashchange", () => { render(); window.scrollTo({ top: 0 }); });

async function start() {
  updateUserChip();
  if (!API_BASE || API_BASE.startsWith("/api")) {
    mount(main, h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚙️"),
      h("div", {}, "The server address isn't set. Open js/config.js and set apiBase to your Hugging Face Space address.")));
    return;
  }
  mount(main, h("div", { class: "boot" }, h("div", { class: "spinner spinner--lg" }), h("p", {}, "Connecting to AnswerMitra…")));
  if (!(await connect())) {
    mount(main, h("div", { class: "boot" }, h("p", {}, "AnswerMitra couldn't connect. Use the Try again button above.")));
    return;
  }
  try {
    await reloadConfig();
  } catch (err) {
    toast(err.message, "error");
    showBanner(err.message, true, h("button", { class: "btn btn--secondary btn--sm", type: "button", onclick: () => start() }, "Try again"));
    return;
  }
  render();
}

start();
