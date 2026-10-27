// My Progress page: sign in / create a profile, then the progress report and past checks.
import { api } from "../api.js";
import { store } from "../store.js";
import { h, icon, ICONS, mount, toast, withBusy, openModal, scorePill, formatDate, kindLabel, emptyState } from "../ui.js";
import { renderResult, reviewBlock } from "../result.js";
import { lineChart } from "../charts.js";

export function renderProgress(root, ctx) {
  if (!store.student()) return renderAuth(root, ctx);
  return renderReport(root, ctx);
}

// ---------- Sign in / create ----------
function renderAuth(root, ctx) {
  const { config } = ctx;
  let mode = "signin";
  const panel = h("div");
  const tabs = h("div", { class: "pill-tabs", role: "tablist" });

  function draw() {
    const nameInput = h("input", { class: "input", id: "p-name", maxlength: "30", autocomplete: "username", placeholder: "e.g. Asha Rao" });
    const pw = h("input", { class: "input", id: "p-pw", type: "password", autocomplete: mode === "signin" ? "current-password" : "new-password" });
    const pw2 = h("input", { class: "input", id: "p-pw2", type: "password", autocomplete: "new-password" });
    const showPw = h("label", { class: "switch small" }, h("input", { type: "checkbox", onchange: (e) => { pw.type = pw2.type = e.target.checked ? "text" : "password"; } }),
      h("span", { class: "switch__track" }), "Show password");
    const submit = h("button", { class: "btn btn--primary btn--block btn--lg", type: "submit" }, mode === "signin" ? "Sign in" : "Create my profile");
    const form = h("form", { novalidate: true, onsubmit: (e) => { e.preventDefault(); go(); } },
      h("div", { class: "field" }, h("label", { class: "label", for: "p-name" }, "Profile name"), nameInput,
        mode === "create" ? h("span", { class: "hint" }, "3 to 30 letters or numbers. No email needed.") : null),
      h("div", { class: "field" }, h("label", { class: "label", for: "p-pw" }, "Password"), pw,
        mode === "create" ? h("span", { class: "hint" }, `At least ${config.limits.min_password} characters.`) : null),
      mode === "create" ? h("div", { class: "field" }, h("label", { class: "label", for: "p-pw2" }, "Type the password again"), pw2) : null,
      h("div", { class: "field" }, showPw),
      submit,
      mode === "signin" ? h("p", { class: "hint", style: { margin: "12px 0 0", textAlign: "center" } }, "Forgot your password? Ask your teacher to set a new one.") : null);

    async function go() {
      const name = nameInput.value.trim();
      if (!name) { nameInput.focus(); return toast("Please type your profile name.", "error"); }
      if (!pw.value) { pw.focus(); return toast("Please type your password.", "error"); }
      if (mode === "create" && pw.value !== pw2.value) { pw2.focus(); return toast("The two passwords don't match. Please type them again.", "error"); }
      await withBusy(submit, mode === "signin" ? "Signing you in…" : "Creating your profile…", async () => {
        try {
          const res = await api(mode === "signin" ? "/profile/login" : "/profile/create", { method: "POST", body: { name, password: pw.value } });
          store.setStudent({ name: res.name, token: res.token });
          toast(mode === "signin" ? `Welcome back, ${res.name}!` : `Profile created! You're signed in as ${res.name}.`, "success");
          ctx.rerender();
        } catch (err) {
          pw.value = ""; pw2.value = "";
          toast(err.message, "error");
        }
      });
    }
    mount(panel, form);
    mount(tabs,
      h("button", { class: "tab", role: "tab", type: "button", "aria-selected": String(mode === "signin"), onclick: () => { mode = "signin"; draw(); } }, "Sign in"),
      h("button", { class: "tab", role: "tab", type: "button", "aria-selected": String(mode === "create"), onclick: () => { mode = "create"; draw(); } }, "Create a profile"));
    nameInput.focus();
  }
  draw();

  mount(root,
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "My Progress 📈"),
      h("p", {}, "Create a free profile to keep your results, see your progress and get your teacher's replies. You only need a name and a password."))),
    h("div", { class: "grid-2" },
      h("section", { class: "card auth-wrap", style: { width: "100%" } }, tabs, panel),
      h("section", { class: "card card--flat" },
        h("h2", {}, "🔒 Your privacy"),
        h("ul", { class: "privacy-list" }, (config.privacy || []).map((line) => h("li", {}, h("span", {}, "✔️"), h("span", {}, line)))),
        h("p", { class: "small muted", style: { margin: "14px 0 0" } }, "You can still check answers without a profile on the Practice page."))));
}

// ---------- Progress report ----------
async function renderReport(root, ctx) {
  const student = store.student();
  const refreshBtn = h("button", { class: "btn btn--secondary btn--sm", type: "button" }, icon(ICONS.refresh), "Refresh");
  const signOut = h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: () => {
    store.setStudent(null); toast("You're signed out. See you next time!", "success"); ctx.rerender();
  } }, icon(ICONS.logout), "Sign out");
  const body = h("div", { class: "stack" }, skeleton());
  mount(root, h("div", { class: "page-head" },
    h("div", {}, h("h1", {}, `${student.name}'s progress 📈`), h("p", {}, "Your checks, scores and the topics to practise next.")),
    h("div", { class: "row" }, refreshBtn, signOut)), body);

  async function load(showToast) {
    try {
      const data = await api("/profile", { auth: "student" });
      mount(body, report(data));
      if (showToast) toast("Your progress is up to date.", "success");
    } catch (err) {
      if (err.status === 401) { toast("Please sign in again.", "warning"); return ctx.rerender(); }
      mount(body, h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚠️"),
        h("div", {}, err.message, h("div", { style: { marginTop: "8px" } }, h("button", { class: "btn btn--secondary btn--sm", type: "button", onclick: () => load(true) }, "Try again")))));
      toast(err.message, "error");
    }
  }
  refreshBtn.addEventListener("click", () => withBusy(refreshBtn, "Refreshing…", () => load(true)));
  await load(false);
}

function skeleton() {
  return h("div", { class: "stack" },
    h("div", { class: "stats" }, [1, 2, 3, 4].map(() => h("div", { class: "skeleton", style: { height: "92px" } }))),
    h("div", { class: "skeleton", style: { height: "260px" } }));
}

function report(data) {
  const p = data.progress;
  if (!p.checks) {
    return h("div", { class: "card" }, emptyState("No checks yet",
      "Go to Practice and check your first answer. Your progress will appear here.",
      h("a", { class: "btn btn--primary", href: "#/practice" }, "Start practising")));
  }
  const trend = p.trend
    ? { up: ["notice--success", "📈", `Improving! Your recent checks average ${p.trend.recent}%, up from ${p.trend.earlier}%. Keep it up!`],
        down: ["notice--warning", "📉", `A little lower lately: ${p.trend.recent}% compared with ${p.trend.earlier}% before. Re-read the pages suggested in your results and try similar questions again. You can do it!`],
        steady: ["", "➡️", `Steady. Your recent checks average ${p.trend.recent}%, about the same as before (${p.trend.earlier}%).`] }[p.trend.status]
    : ["", "ℹ️", "Make at least 4 checks to see how you're improving."];

  const groups = {};
  p.groups.forEach((g) => (groups[g.group] = groups[g.group] || []).push(g));

  return h("div", { class: "stack" },
    h("div", { class: "stats" },
      stat("Checks made", p.checks, p.checks - p.scored ? `${p.checks - p.scored} not scored` : ""),
      stat("Average", p.average !== null ? `${p.average}%` : "–"),
      stat("Best", p.best !== null ? `${p.best}%` : "–"),
      stat("Latest", p.latest !== null ? `${p.latest}%` : "–", p.teacher_marked ? `${p.teacher_marked} marked by teacher` : "")),
    h("div", { class: `notice ${trend[0]}` }, h("span", { class: "notice__icon" }, trend[1]), h("div", {}, trend[2])),
    p.chart.length >= 2 ? h("section", { class: "card" }, h("h2", {}, "Your scores over time"), lineChart(p.chart)) : null,
    h("div", { class: "grid-2" },
      h("section", { class: "card" }, h("h2", {}, "Averages"),
        h("div", { class: "bars" }, Object.entries(groups).map(([group, items]) => [
          h("div", { class: "group-label" }, group),
          items.map((g) => h("div", { class: "bar-row" }, h("span", { class: "name", title: g.name }, g.name),
            h("div", { class: "bar", role: "img", "aria-label": `${g.average} percent` }, h("span", { style: { width: `${g.average}%` } })),
            h("span", { class: "val" }, `${g.average}%`)))]))),
      h("section", { class: "card" }, h("h2", {}, "Book topics"),
        h("p", { class: "label", style: { margin: "4px 0 6px" } }, "💪 Strong (75% or more)"),
        p.strong.length ? h("div", { class: "chips" }, p.strong.map((t) => h("span", { class: "badge badge--success" }, `${t.topic} · ${t.average}%`)))
          : h("p", { class: "small muted" }, "None yet. Keep practising!"),
        h("p", { class: "label", style: { margin: "14px 0 6px" } }, "🎯 Practise next (under 60%)"),
        p.practise.length ? h("div", { class: "chips" }, p.practise.map((t) => h("span", { class: "badge badge--accent" }, `${t.topic} · ${t.average}%`)))
          : h("p", { class: "small muted" }, "Nothing urgent. Great work!"))),
    history(data.checks));
}

function stat(label, value, sub = "") {
  return h("div", { class: "stat" }, h("div", { class: "stat__label" }, label), h("div", { class: "stat__value" }, value),
    sub ? h("div", { class: "stat__sub" }, sub) : null);
}

function history(checks) {
  let filter = "all";
  let query = "";
  const list = h("div", { class: "list" });
  const filters = [["all", "All"], ["reviewed", "Teacher marked"], ["hots", "HOTS"], ["photo", "📷 Photo"]];
  const filterBar = h("div", { class: "pill-tabs", role: "tablist", "aria-label": "Filter checks" });
  const search = h("input", { class: "input", type: "search", placeholder: "Search your questions…", "aria-label": "Search your questions",
    oninput: (e) => { query = e.target.value.toLowerCase(); draw(); } });

  function draw() {
    mount(filterBar, filters.map(([key, label]) => h("button", { class: "tab", role: "tab", type: "button", "aria-selected": String(filter === key),
      onclick: () => { filter = key; draw(); } }, label)));
    const shown = checks.filter((c) => (filter === "all" || (filter === "reviewed" && c.review_status === "reviewed")
      || (filter === "hots" && c.question_kind === "hots") || (filter === "photo" && c.answer_source === "photo"))
      && (!query || c.question.toLowerCase().includes(query)));
    mount(list, shown.length ? shown.map((c) => h("button", { class: "list-item", type: "button", onclick: () => openCheck(c) },
      h("div", { class: "list-item__main" },
        h("div", { class: "list-item__title" }, c.question),
        h("div", { class: "list-item__meta" }, h("span", {}, formatDate(c.submitted_at)), h("span", {}, "·"), h("span", {}, c.question_type),
          h("span", { class: "badge" }, kindLabel(c.question_kind)),
          c.answer_source === "photo" ? h("span", { class: "badge" }, "📷") : null,
          c.review_status === "reviewed" ? h("span", { class: "badge badge--primary" }, "👩‍🏫 Teacher marked")
            : c.review_status === "requested" ? h("span", { class: "badge badge--warning" }, "⏳ Review requested") : null)),
      scorePill(c.final_percent)))
      : h("p", { class: "muted small", style: { margin: "8px 0" } }, "No checks match."));
  }
  draw();
  return h("section", { class: "card" }, h("div", { class: "row row--between" }, h("h2", { style: { margin: 0 } }, "Previous checks"),
    h("span", { class: "muted small" }, `${checks.length} total`)),
    h("div", { class: "toolbar", style: { marginTop: "12px" } }, search), filterBar, list);
}

async function openCheck(summary) {
  const body = h("div", {}, h("div", { class: "skeleton", style: { height: "200px" } }));
  openModal(summary.question, body);
  try {
    const check = await api(`/profile/checks/${encodeURIComponent(summary.id)}`, { auth: "student" });
    mount(body,
      h("p", { class: "small muted", style: { marginTop: 0 } }, `${check.book_title} · ${formatDate(check.submitted_at)}`),
      h("h3", {}, "Your answer"), h("div", { class: "model-answer", style: { marginBottom: "14px" } }, check.answer),
      reviewBlock(check.review, { aiPercent: check.ai_percent }),
      check.review?.teacher_percent !== null && check.review?.teacher_percent !== undefined
        ? h("h3", { style: { marginTop: "14px" } }, "What the AI said") : null,
      h("div", { style: { marginTop: "14px" } }, renderResult({ ...check, final_percent: check.ai_percent }, { compact: true })));
  } catch (err) {
    mount(body, h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚠️"), h("div", {}, err.message)));
    toast(err.message, "error");
  }
}
