// Shows a checked answer: score ring, what went well, what to add, model answer and pages.
import { h, icon, ICONS, kindLabel, toast } from "./ui.js";

const RING_R = 48;
const RING_C = 2 * Math.PI * RING_R;

function ring(percent) {
  const value = Math.max(0, Math.min(100, percent ?? 0));
  const color = value >= 75 ? "var(--success)" : value >= 50 ? "var(--accent)" : "var(--danger)";
  const fg = h("circle", { class: "ring-fg", cx: "56", cy: "56", r: String(RING_R), stroke: color,
    "stroke-dasharray": String(RING_C), "stroke-dashoffset": String(RING_C) });
  // Animate the ring filling up after it is on screen.
  requestAnimationFrame(() => requestAnimationFrame(() =>
    fg.setAttribute("stroke-dashoffset", String(RING_C * (1 - value / 100)))));
  return h("div", { class: "ring", role: "img", "aria-label": `Score ${value} percent` },
    h("svg", { viewBox: "0 0 112 112" }, h("circle", { class: "ring-bg", cx: "56", cy: "56", r: String(RING_R) }), fg),
    h("div", { class: "ring__label" }, h("div", {}, h("strong", {}, `${value}%`), h("small", {}, "score"))));
}

function pointList(points, iconFor) {
  return h("ul", { class: "point-list" }, points.map((p) =>
    h("li", {}, h("span", { class: "point-icon", "aria-hidden": "true" }, iconFor(p)),
      h("div", {}, h("div", { class: "pt" }, p.point), p.comment ? h("div", { class: "cm" }, p.comment) : null))));
}

/** The teacher's review of a check, if there is one. */
export function reviewBlock(review, { forTeacher = false, aiPercent } = {}) {
  if (!review) return null;
  const parts = [];
  if (review.reason !== null && review.reason !== undefined) {
    parts.push(h("p", { class: "small" }, h("strong", {}, forTeacher ? "Student's reason: " : "You asked your teacher to review this: "),
      review.reason || "(no reason given)"));
  }
  if (review.teacher_percent !== null && review.teacher_percent !== undefined) {
    parts.push(h("div", { class: "row" }, h("span", { class: "badge badge--primary" }, "👩‍🏫 Teacher's mark"),
      h("strong", {}, `${review.teacher_percent}%`),
      aiPercent !== null && aiPercent !== undefined ? h("span", { class: "muted small" }, `(the AI said ${aiPercent}%)`) : null));
    if (review.teacher_comment) parts.push(h("div", { class: "model-answer", style: { marginTop: "8px", borderLeftColor: "var(--accent)" } }, review.teacher_comment));
  } else if (!forTeacher) {
    parts.push(h("p", { class: "small muted", style: { margin: 0 } }, "⏳ Waiting for your teacher to review this check."));
  }
  return h("div", { class: "notice notice--warning", style: { flexDirection: "column", gap: "6px" } }, parts);
}

/**
 * Render a check. options.actions: extra elements under the result (buttons, review form).
 */
export function renderResult(check, { actions = null, compact = false } = {}) {
  const r = check.result || {};
  if (!r.covered_by_book) {
    return h("div", { class: "result" },
      h("div", { class: "notice notice--warning" }, h("span", { class: "notice__icon" }, "📖"),
        h("div", {}, h("strong", {}, "This question doesn't seem to be related to the book."),
          h("p", { style: { margin: "4px 0 0" } }, r.overall_feedback || ""),
          h("p", { class: "small muted", style: { margin: "4px 0 0" } },
            "Try a question about a topic from the book. HOTS questions are welcome if they use the book's ideas."))),
      actions);
  }
  const points = r.key_points || [];
  const covered = points.filter((p) => p.status === "covered");
  const partial = points.filter((p) => p.status === "partial");
  const missing = points.filter((p) => p.status === "missing");
  const headline = covered.length === points.length
    ? `🌟 You covered all ${points.length} key points!`
    : `🌱 You covered ${covered.length} of ${points.length} key points`;

  const copyBtn = h("button", { class: "btn btn--ghost btn--sm", type: "button", onclick: async () => {
    try { await navigator.clipboard.writeText(r.correct_answer || ""); toast("Model answer copied.", "success"); }
    catch { toast("Couldn't copy. Select the text and copy it yourself.", "error"); }
  } }, icon(ICONS.copy), "Copy");

  return h("div", { class: "result" },
    h("div", { class: "result-head" },
      ring(check.final_percent ?? r.correct_percent),
      h("div", {},
        h("h2", {}, headline),
        partial.length ? h("p", { class: "muted", style: { margin: "0 0 8px" } }, `…and partly covered ${partial.length} more.`) : null,
        h("div", { class: "chips" },
          check.question_type ? h("span", { class: "chip" }, check.question_type) : null,
          h("span", { class: "chip" }, kindLabel(check.question_kind)),
          check.answer_source === "photo" ? h("span", { class: "chip" }, "📷 From photo") : null,
          check.book_title ? h("span", { class: "chip" }, "📘 " + check.book_title) : null))),
    r.concepts_used?.length ? h("div", { class: "row small" }, h("strong", {}, "Book concepts:"),
      h("div", { class: "chips" }, r.concepts_used.map((c) => h("span", { class: "badge badge--primary" }, c)))) : null,
    h("section", { class: "section section--good" }, h("h3", {}, "✅ What you did well"),
      covered.length ? pointList(covered, () => "✔️")
        : h("p", { style: { margin: 0 } }, "You've made a start, and checking your answer is how you learn. Look at the points below.")),
    partial.length || missing.length ? h("section", { class: "section section--improve" },
      h("h3", {}, `➕ Add ${partial.length + missing.length === 1 ? "this" : "these"} to improve`),
      pointList([...partial, ...missing], (p) => (p.status === "partial" ? "🟡" : "➕"))) : null,
    r.incorrect_statements?.length ? h("section", { class: "section section--fix" }, h("h3", {}, "🔁 Check these against the book"),
      h("ul", { class: "point-list" }, r.incorrect_statements.map((s) => h("li", {}, h("span", { "aria-hidden": "true" }, "•"), h("div", {}, s))))) : null,
    r.improvements?.length ? h("section", { class: "section" }, h("h3", {}, "💡 Tips for next time"),
      h("ul", { class: "point-list" }, r.improvements.map((s) => h("li", {}, h("span", { "aria-hidden": "true" }, "→"), h("div", {}, s))))) : null,
    r.correct_answer ? h("section", { class: "section" },
      h("details", { class: "disclosure", open: !compact },
        h("summary", {}, "📘 A model answer (based on the book)"),
        h("div", {}, h("div", { class: "model-answer" }, r.correct_answer), h("div", { class: "row", style: { marginTop: "8px" } }, copyBtn)))) : null,
    r.relevant_pages?.length ? h("div", { class: "row" }, h("strong", {}, "📖 Read again:"),
      h("div", { class: "chips" }, r.relevant_pages.map((p) => h("span", { class: "chip" }, `Page ${p}`)))) : null,
    r.overall_feedback ? h("div", { class: "notice notice--success" }, h("span", { class: "notice__icon" }, "💪"),
      h("div", {}, h("strong", {}, "Keep going: "), r.overall_feedback)) : null,
    h("p", { class: "small muted", style: { margin: 0 } },
      "How the score works: points you covered fully count in full, partly covered points count half, and each statement that disagrees with the book takes off 10%."),
    actions);
}

/** Animated "working on it" steps shown while a check runs. */
export function loadingSteps() {
  const labels = ["Finding the right pages in your book", "Comparing your answer with the book", "Writing your feedback"];
  const steps = labels.map((label) => h("div", { class: "loading-step" }, h("span", { class: "dot" }), label));
  let index = 0;
  const advance = () => {
    steps.forEach((s, i) => {
      s.classList.toggle("is-done", i < index);
      s.classList.toggle("is-active", i === index);
      s.querySelector(".dot").textContent = i < index ? "✓" : "";
    });
  };
  advance();
  const timers = [setTimeout(() => { index = 1; advance(); }, 3500), setTimeout(() => { index = 2; advance(); }, 8000)];
  const el = h("div", { class: "card", "aria-live": "polite" },
    h("div", { class: "card__title" }, h("span", { class: "spinner" }), h("h3", {}, "Checking your answer…")),
    h("div", { class: "loading-steps" }, steps),
    h("p", { class: "small muted", style: { margin: "12px 0 0" } }, "This usually takes 5 to 20 seconds. Please keep this page open."));
  el.stop = () => timers.forEach(clearTimeout);
  return el;
}
