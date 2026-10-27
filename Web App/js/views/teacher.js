// Teacher dashboard: reviews and marks, results, books and student profiles.
import { api, uploadFiles, downloadFile } from "../api.js";
import { store } from "../store.js";
import { h, icon, ICONS, mount, toast, withBusy, openModal, scorePill, formatDate, kindLabel, emptyState } from "../ui.js";
import { renderResult, reviewBlock } from "../result.js";

let activeTab = "reviews";
let flash = "";   // a short "saved" note shown after the dashboard refreshes

export function renderTeacher(root, ctx) {
  if (!store.teacherToken()) return renderLogin(root, ctx);
  return renderDashboard(root, ctx);
}

// ---------- Login ----------
function renderLogin(root, ctx) {
  const pw = h("input", { class: "input", id: "t-pw", type: "password", autocomplete: "current-password" });
  const submit = h("button", { class: "btn btn--primary btn--block btn--lg", type: "submit" }, "Log in");
  const form = h("form", { onsubmit: (e) => { e.preventDefault(); go(); } },
    h("div", { class: "field" }, h("label", { class: "label", for: "t-pw" }, "Teacher password"), pw), submit,
    h("p", { class: "hint", style: { margin: "12px 0 0", textAlign: "center" } },
      "You stay logged in until you close this browser tab, which keeps shared classroom computers safe."));
  async function go() {
    if (!pw.value) { pw.focus(); return toast("Please type the teacher password.", "error"); }
    await withBusy(submit, "Logging in…", async () => {
      try {
        const res = await api("/teacher/login", { method: "POST", body: { password: pw.value } });
        store.setTeacherToken(res.token);
        toast("Logged in as teacher.", "success");
        ctx.rerender();
      } catch (err) {
        pw.value = "";
        pw.focus();
        toast(err.message, "error");
      }
    });
  }
  mount(root, h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Teacher 👩‍🏫"),
      h("p", {}, "Upload books, answer review requests, mark answers and see how your class is doing."))),
    h("section", { class: "card auth-wrap" }, form));
  pw.focus();
}

// ---------- Dashboard ----------
async function renderDashboard(root, ctx) {
  const refreshBtn = h("button", { class: "btn btn--secondary btn--sm", type: "button" }, icon(ICONS.refresh), "Refresh");
  const logout = h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: () => {
    store.setTeacherToken(null); toast("Logged out.", "success"); ctx.rerender();
  } }, icon(ICONS.logout), "Log out");
  const body = h("div", { class: "stack" }, h("div", { class: "skeleton", style: { height: "120px" } }), h("div", { class: "skeleton", style: { height: "320px" } }));
  mount(root, h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Teacher dashboard 👩‍🏫"),
    h("p", {}, "Your class at a glance. Students' review requests appear under Reviews.")), h("div", { class: "row" }, refreshBtn, logout)), body);

  const state = { data: null };
  async function load(showToast) {
    try {
      state.data = await api("/teacher/dashboard", { auth: "teacher" });
      draw();
      if (showToast) toast("Dashboard is up to date.", "success");
    } catch (err) {
      if (err.status === 401) { toast("Please log in again.", "warning"); return ctx.rerender(); }
      mount(body, h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚠️"), h("div", {}, err.message)));
      toast(err.message, "error");
    }
  }
  refreshBtn.addEventListener("click", () => withBusy(refreshBtn, "Refreshing…", () => load(true)));

  function draw() {
    const d = state.data;
    const a = d.agreement;
    const tabs = [["reviews", "Reviews", d.reviews_waiting.length], ["results", "Results"], ["books", "Books", d.books.length], ["students", "Students", d.profiles.length]];
    const panel = h("div");
    const tabBar = h("div", { class: "tabs", role: "tablist", "aria-label": "Dashboard sections" });
    const select = (key) => {
      activeTab = key;
      mount(tabBar, tabs.map(([k, label, count]) => h("button", { class: "tab", role: "tab", type: "button", "aria-selected": String(k === key), onclick: () => select(k) },
        label, count !== undefined ? h("span", { class: `badge ${k === "reviews" && count ? "badge--warning" : ""}` }, String(count)) : null)));
      const views = { reviews: reviewsTab, results: resultsTab, books: booksTab, students: studentsTab };
      views[key](panel, d, { reload: () => load(false), ctx });
    };
    mount(body,
      d.setup_problems.length ? h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚠️"),
        h("div", {}, h("strong", {}, "Setup problems: "), d.setup_problems.join(" "))) : null,
      d.unsynced ? h("div", { class: "notice notice--warning" }, h("span", { class: "notice__icon" }, "⏳"),
        h("div", {}, `${d.unsynced} saved file(s) are still being copied to your Dataset. They will finish automatically.`)) : null,
      h("div", { class: "stats" },
        stat("Checks", d.stats.checks, `${d.stats.photos} from photos · ${d.stats.hots} HOTS`),
        stat("Students", d.stats.students, `${d.profiles.length} with a profile`),
        stat("Class average", d.stats.average !== null ? `${d.stats.average}%` : "–"),
        stat("Waiting for review", d.reviews_waiting.length),
        stat("AI vs teacher", a.marked ? `${a.within}/${a.marked}` : "–",
          a.marked ? `within ${a.tolerance} points · avg diff ${a.average_difference}` : "Mark checks to measure")),
      h("section", { class: "card" }, tabBar, panel));
    select(activeTab);
  }
  await load(false);
}

function stat(label, value, sub = "") {
  return h("div", { class: "stat" }, h("div", { class: "stat__label" }, label), h("div", { class: "stat__value" }, String(value)),
    sub ? h("div", { class: "stat__sub" }, sub) : null);
}

// ---------- Marking panel (shared by Reviews and Results) ----------
async function markingPanel(container, checkId, { onSaved } = {}) {
  mount(container, h("div", { class: "skeleton", style: { height: "260px" } }));
  let check;
  try {
    check = await api(`/teacher/checks/${encodeURIComponent(checkId)}`, { auth: "teacher" });
  } catch (err) {
    mount(container, h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚠️"), h("div", {}, err.message)));
    return toast(err.message, "error");
  }
  const start = check.review?.teacher_percent ?? check.ai_percent ?? 0;
  const range = h("input", { type: "range", min: "0", max: "100", step: "1", value: String(start), "aria-label": "Your mark" });
  const number = h("input", { class: "input", type: "number", min: "0", max: "100", value: String(start), "aria-label": "Your mark in percent" });
  range.addEventListener("input", () => (number.value = range.value));
  number.addEventListener("input", () => (range.value = number.value));
  const comment = h("textarea", { class: "textarea", rows: "3", maxlength: "1000", value: check.review?.teacher_comment || "",
    placeholder: "Optional comment for the student…" });
  const save = h("button", { class: "btn btn--primary", type: "button" }, icon(ICONS.check), "Save my mark");
  save.addEventListener("click", () => withBusy(save, "Saving your mark…", async () => {
    const mark = Number(number.value);
    if (number.value === "" || Number.isNaN(mark) || mark < 0 || mark > 100) { number.focus(); return toast("Please enter your mark from 0 to 100.", "error"); }
    try {
      const res = await api(`/teacher/checks/${encodeURIComponent(checkId)}/mark`, { method: "POST", auth: "teacher",
        body: { mark, comment: comment.value.trim() } });
      toast(res.message, "success");
      onSaved?.(res.check);
      markingPanel(container, checkId, { onSaved });
    } catch (err) {
      toast(err.message, "error");
    }
  }));
  mount(container,
    h("div", { class: "row row--between" }, h("div", {}, h("strong", {}, check.student_name),
      h("span", { class: "muted small" }, ` · ${check.book_title} · ${formatDate(check.submitted_at)}`)),
      h("div", { class: "chips" }, h("span", { class: "chip" }, check.question_type), h("span", { class: "chip" }, kindLabel(check.question_kind)),
        check.answer_source === "photo" ? h("span", { class: "chip" }, "📷 Photo") : null,
        check.has_profile ? h("span", { class: "badge badge--primary" }, "Has profile") : h("span", { class: "badge" }, "Guest"))),
    h("h3", { style: { marginTop: "12px" } }, "Question"), h("p", {}, check.question),
    h("h3", {}, "Answer"), h("div", { class: "model-answer", style: { marginBottom: "14px" } }, check.answer),
    reviewBlock(check.review, { forTeacher: true, aiPercent: check.ai_percent }),
    h("div", { class: "card card--flat", style: { margin: "14px 0", background: "var(--surface-2)" } },
      h("h3", {}, "👩‍🏫 Your mark"),
      h("div", { class: "range-row field" }, range, number, h("span", { class: "muted" }, "%")),
      h("div", { class: "field" }, comment),
      h("div", { class: "row" }, save, h("span", { class: "small muted" }, `The AI gave ${check.ai_percent ?? "–"}%.`,
        check.has_profile ? " The student will see your mark in My Progress." : " This student wasn't signed in, so they can't see it in the app."))),
    h("details", { class: "disclosure" }, h("summary", {}, "What the AI said"),
      h("div", {}, renderResult({ ...check, final_percent: check.ai_percent }, { compact: true }))));
}

// ---------- Tabs ----------
function reviewsTab(panel, d, { reload }) {
  if (!d.reviews_waiting.length) {
    const note = flash;
    flash = "";
    return mount(panel, note ? h("div", { class: "notice notice--success", style: { marginBottom: "12px" } },
      h("span", { class: "notice__icon" }, "✅"), h("div", {}, note)) : null, emptyState("No reviews waiting 🎉",
      "When a student presses “Teacher, please review”, it appears here. You can also mark any check from the Results tab."));
  }
  const detail = h("div", {}, flash
    ? h("div", { class: "notice notice--success" }, h("span", { class: "notice__icon" }, "✅"), h("div", {}, flash, " Choose the next check on the left."))
    : h("p", { class: "muted" }, "Choose a check on the left to review it."));
  flash = "";
  let selected = null;
  const list = h("div", { class: "list" });
  const drawList = () => mount(list, d.reviews_waiting.map((c) => {
    const item = h("button", { class: `list-item ${selected === c.id ? "is-selected" : ""}`, type: "button", onclick: () => {
      selected = c.id; drawList();
      markingPanel(detail, c.id, { onSaved: (saved) => { flash = `Saved your mark of ${saved.review.teacher_percent}% for ${saved.student_name}.`; reload(); } });
      if (window.matchMedia("(max-width: 900px)").matches) detail.scrollIntoView({ behavior: "smooth" });
    } },
      h("div", { class: "list-item__main" }, h("div", { class: "list-item__title" }, c.student_name),
        h("div", { class: "list-item__meta" }, h("span", {}, c.question.slice(0, 60)), h("span", {}, formatDate(c.submitted_at))),
        c.reason ? h("div", { class: "small muted", style: { marginTop: "4px" } }, `“${c.reason.slice(0, 80)}”`) : null),
      scorePill(c.ai_percent));
    return item;
  }));
  drawList();
  mount(panel, h("div", { class: "split" }, h("div", {}, list), h("div", {}, detail)));
}

async function resultsTab(panel, d) {
  const bookFilter = h("select", { class: "select", "aria-label": "Filter by book" },
    h("option", { value: "" }, "All books"), d.books.map((b) => h("option", { value: b.id }, b.title)));
  const search = h("input", { class: "input", type: "search", placeholder: "Search student or question…", "aria-label": "Search results" });
  const csvBtn = h("button", { class: "btn btn--secondary", type: "button" }, icon(ICONS.download), "Download CSV");
  const tableArea = h("div", {}, h("div", { class: "skeleton", style: { height: "200px" } }));
  let rows = [];
  csvBtn.addEventListener("click", () => withBusy(csvBtn, "Preparing…", async () => {
    try {
      await downloadFile(`/teacher/results.csv${bookFilter.value ? `?book=${encodeURIComponent(bookFilter.value)}` : ""}`, "answermitra_results.csv");
      toast("CSV downloaded.", "success");
    } catch (err) { toast(err.message, "error"); }
  }));
  async function load() {
    mount(tableArea, h("div", { class: "skeleton", style: { height: "200px" } }));
    try {
      const res = await api(`/teacher/results${bookFilter.value ? `?book=${encodeURIComponent(bookFilter.value)}` : ""}`, { auth: "teacher" });
      rows = res.rows;
      draw();
    } catch (err) {
      mount(tableArea, h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚠️"), h("div", {}, err.message)));
      toast(err.message, "error");
    }
  }
  function draw() {
    const q = search.value.toLowerCase();
    const shown = rows.filter((r) => !q || r.student_name.toLowerCase().includes(q) || r.question.toLowerCase().includes(q));
    if (!shown.length) return mount(tableArea, emptyState("No checks yet", q ? "Nothing matches your search." : "Checks will appear here as students use AnswerMitra."));
    mount(tableArea, h("p", { class: "small muted" }, `${shown.length} check(s). Click a row to see it and give your own mark.`),
      h("div", { class: "table-wrap" }, h("table", { class: "table" },
        h("thead", {}, h("tr", {}, ["Student", "Question", "Book", "Type", "AI", "Teacher", "Review", "When"].map((t) => h("th", { scope: "col" }, t)))),
        h("tbody", {}, shown.map((r) => h("tr", { tabindex: "0", onclick: () => openRow(r),
          onkeydown: (e) => e.key === "Enter" && openRow(r) },
          h("td", { class: "nowrap" }, r.student_name, r.has_profile ? " ✅" : ""),
          h("td", { class: "clip", title: r.question }, r.question),
          h("td", { class: "clip" }, r.book_title),
          h("td", { class: "nowrap" }, `${r.question_type} · ${kindLabel(r.question_kind)}${r.answer_source === "photo" ? " · 📷" : ""}`),
          h("td", {}, scorePill(r.ai_percent)),
          h("td", {}, r.teacher_percent !== null && r.teacher_percent !== undefined ? scorePill(r.teacher_percent) : "–"),
          h("td", {}, r.review_status === "requested" ? h("span", { class: "badge badge--warning" }, "Requested")
            : r.review_status === "reviewed" ? h("span", { class: "badge badge--success" }, "Reviewed") : ""),
          h("td", { class: "nowrap muted" }, formatDate(r.submitted_at))))))));
  }
  function openRow(r) {
    const container = h("div");
    openModal(`${r.student_name}: ${r.question.slice(0, 50)}`, container);
    markingPanel(container, r.id, { onSaved: load });
  }
  bookFilter.addEventListener("change", load);
  search.addEventListener("input", draw);
  mount(panel, h("div", { class: "toolbar" }, bookFilter, search, csvBtn), tableArea);
  await load();
}

function booksTab(panel, d, { reload, ctx }) {
  let files = [];
  const fileInput = h("input", { type: "file", accept: "application/pdf,.pdf", multiple: true, class: "sr-only", id: "book-files",
    onchange: (e) => setFiles([...e.target.files]) });
  const chosen = h("div", { class: "small muted" });
  const title = h("input", { class: "input", placeholder: "Book title (optional, only for a single book)", "aria-label": "Book title", disabled: true });
  const progress = h("div", { class: "bar", hidden: true }, h("span", { style: { width: "0%" } }));
  const resultsBox = h("div");
  const uploadBtn = h("button", { class: "btn btn--primary", type: "button", disabled: true }, icon(ICONS.upload), "Upload");
  function setFiles(list) {
    files = list.filter((f) => f.name.toLowerCase().endsWith(".pdf"));
    if (list.length && !files.length) toast("Please choose PDF files.", "error");
    chosen.textContent = files.length ? `${files.length} file(s): ${files.map((f) => f.name).join(", ")}` : "";
    title.disabled = files.length !== 1;
    uploadBtn.disabled = !files.length;
  }
  const zone = h("label", { class: "dropzone", for: "book-files", tabindex: "0",
    onkeydown: (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), fileInput.click()),
    ondragover: (e) => { e.preventDefault(); zone.classList.add("is-over"); },
    ondragleave: () => zone.classList.remove("is-over"),
    ondrop: (e) => { e.preventDefault(); zone.classList.remove("is-over"); setFiles([...e.dataTransfer.files]); } },
    icon(ICONS.book), h("strong", {}, "Drop PDF books here or click to choose"), h("span", { class: "hint" }, "You can choose several books at once. Scanned PDFs (pictures of pages) can't be read."));
  uploadBtn.addEventListener("click", () => withBusy(uploadBtn, "Uploading…", async () => {
    progress.hidden = false;
    const bar = progress.firstChild;
    toast("Uploading and reading your PDF(s). A big book can take up to a minute.", "warning");
    try {
      const res = await uploadFiles("/teacher/books", files, { title: files.length === 1 ? title.value.trim() : "" },
        (fraction) => { bar.style.width = `${Math.round(fraction * 100)}%`; });
      const ok = res.results.filter((r) => r.ok).length;
      mount(resultsBox, h("div", { class: "list" }, res.results.map((r) => h("div", { class: `notice ${r.ok ? "notice--success" : "notice--danger"}` },
        h("span", { class: "notice__icon" }, r.ok ? "✅" : "❌"), h("div", {}, r.message)))));
      if (ok) toast(`${ok} book(s) uploaded. Students can choose them now.`, "success");
      if (ok < res.results.length) toast(`${res.results.length - ok} file(s) couldn't be used. See the details below.`, "error");
      files = []; fileInput.value = ""; setFiles([]); title.value = "";
      await ctx.reloadConfig();
      reload();
    } catch (err) {
      toast(err.message, "error");
    } finally {
      progress.hidden = true;
      bar.style.width = "0%";
    }
  }));

  const list = d.books.length ? h("div", { class: "list" }, d.books.map((b) => {
    const toggle = h("input", { type: "checkbox", checked: b.visible });
    toggle.addEventListener("change", async () => {
      toggle.disabled = true;
      try {
        const res = await api(`/teacher/books/${encodeURIComponent(b.id)}/visibility`, { method: "POST", auth: "teacher", body: { visible: toggle.checked } });
        toast(res.message, "success");
        await ctx.reloadConfig();
      } catch (err) {
        toggle.checked = !toggle.checked;
        toast(err.message, "error");
      } finally { toggle.disabled = false; }
    });
    return h("div", { class: "list-item", style: { cursor: "default" } },
      h("div", { class: "list-item__main" }, h("div", { class: "list-item__title" }, b.title),
        h("div", { class: "list-item__meta" }, h("span", {}, `${b.pages} pages`), h("span", {}, `Uploaded ${formatDate(b.uploaded, false)}`))),
      h("label", { class: "switch" }, toggle, h("span", { class: "switch__track" }), h("span", {}, "Students can see it")));
  })) : emptyState("No books yet", "Upload your first book above.");

  mount(panel, h("div", { class: "stack" },
    h("div", { class: "stack" }, fileInput, zone, chosen, title, progress, h("div", { class: "row" }, uploadBtn), resultsBox),
    h("h3", { style: { margin: "8px 0 0" } }, "Your books"), list));
}

function studentsTab(panel, d) {
  const search = h("input", { class: "input", type: "search", placeholder: "Search students…", "aria-label": "Search students" });
  const list = h("div", { class: "list" });
  function draw() {
    const q = search.value.toLowerCase();
    const shown = d.profiles.filter((p) => !q || p.name.toLowerCase().includes(q));
    mount(list, shown.length ? shown.map((p) => {
      const btn = h("button", { class: "btn btn--secondary btn--sm", type: "button", onclick: () => resetPassword(p) }, "Set new password");
      return h("div", { class: "list-item", style: { cursor: "default" } },
        h("div", { class: "list-item__main" }, h("div", { class: "list-item__title" }, p.name),
          h("div", { class: "list-item__meta" }, h("span", {}, `${p.checks} check(s)`), h("span", {}, `Joined ${formatDate(p.created, false)}`))), btn);
    }) : emptyState(d.profiles.length ? "No match" : "No profiles yet", d.profiles.length ? "" : "Students create profiles on the My Progress page."));
  }
  search.addEventListener("input", draw);
  draw();
  mount(panel, h("p", { class: "small muted", style: { marginTop: 0 } }, "Passwords are never shown, not even to you. If a student forgets theirs, set a new one and tell them."),
    h("div", { class: "toolbar" }, search), list);
}

function resetPassword(profile) {
  const input = h("input", { class: "input", type: "text", minlength: "6", maxlength: "128", autocomplete: "off", placeholder: "At least 6 characters" });
  const save = h("button", { class: "btn btn--primary", type: "button" }, "Save new password");
  let close;
  save.addEventListener("click", () => withBusy(save, "Saving…", async () => {
    try {
      const res = await api(`/teacher/profiles/${encodeURIComponent(profile.key)}/password`, { method: "POST", auth: "teacher", body: { password: input.value } });
      toast(res.message, "success");
      close();
    } catch (err) { toast(err.message, "error"); }
  }));
  close = openModal(`New password for ${profile.name}`, h("div", {},
    h("p", { class: "muted", style: { marginTop: 0 } }, "Type a new password, save it, and tell the student. Their old password stops working, and they'll need to sign in again."),
    h("div", { class: "field" }, h("label", { class: "label" }, "New password"), input), save));
  input.focus();
}
