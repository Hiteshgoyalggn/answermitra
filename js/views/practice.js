// Practice page: choose a book, type or photograph an answer, and get feedback.
import { api, ApiError } from "../api.js";
import { store } from "../store.js";
import { h, icon, ICONS, mount, toast, withBusy, counter, segmented } from "../ui.js";
import { renderResult, loadingSteps } from "../result.js";

/** Shrink a photo in the browser before upload (faster on school Wi-Fi). Returns a JPEG data URL. */
async function shrinkPhoto(file, maxSide) {
  const bitmap = await createImageBitmap(file).catch(() => null);
  if (!bitmap) throw new Error("unreadable");
  const scale = Math.min(1, maxSide / Math.max(bitmap.width, bitmap.height));
  const canvas = Object.assign(document.createElement("canvas"), {
    width: Math.round(bitmap.width * scale), height: Math.round(bitmap.height * scale) });
  canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height);
  bitmap.close?.();
  return canvas.toDataURL("image/jpeg", 0.85);
}

export function renderPractice(root, ctx) {
  const { config } = ctx;
  const prefs = store.prefs();
  const draft = store.draft();
  const student = store.student();
  const books = config.books || [];
  const state = {
    bookId: books.some((b) => b.id === prefs.bookId) ? prefs.bookId : books[0]?.id || "",
    questionType: prefs.questionType || config.default_question_type,
    language: prefs.language || config.default_language,
    answerSource: draft.answerSource || "typed",
    photoData: null,
  };

  // ----- Inputs -----
  const nameInput = h("input", { class: "input", id: "name", maxlength: String(config.limits.name), autocomplete: "name",
    value: draft.name || "", placeholder: "e.g. Asha Rao", oninput: (e) => store.setDraft({ name: e.target.value }) });
  const bookSelect = h("select", { class: "select", id: "book", onchange: (e) => { state.bookId = e.target.value; store.setPrefs({ bookId: state.bookId }); } },
    books.map((b) => h("option", { value: b.id, selected: b.id === state.bookId }, `${b.title} (${b.pages} pages)`)));
  const questionInput = h("textarea", { class: "textarea", id: "question", rows: "3", maxlength: String(config.limits.question),
    value: draft.question || "", placeholder: "Type the question you are practising…",
    oninput: (e) => store.setDraft({ question: e.target.value }) });
  const answerInput = h("textarea", { class: "textarea textarea--answer", id: "answer", maxlength: String(config.limits.answer),
    value: draft.answer || "", placeholder: "Write your answer here, the way you would in the exam…",
    oninput: (e) => { store.setDraft({ answer: e.target.value }); unclearNote.hidden = !/\[unclear\]/i.test(e.target.value); } });
  const unclearNote = h("div", { class: "notice notice--warning", hidden: !/\[unclear\]/i.test(answerInput.value) },
    h("span", { class: "notice__icon" }, "✏️"),
    h("div", {}, "Some words in your photo couldn't be read and are marked ", h("mark", { class: "unclear" }, "[unclear]"),
      ". Please type the right words before checking."));

  // ----- Photo -----
  const fileInput = h("input", { type: "file", accept: "image/*", capture: "environment", class: "sr-only", id: "photo-file",
    onchange: (e) => e.target.files[0] && choosePhoto(e.target.files[0]) });
  const photoArea = h("div");
  const readBtn = h("button", { class: "btn btn--primary btn--block", type: "button", onclick: readPhoto }, icon(ICONS.sparkle), "Read my photo");
  function dropzone() {
    const zone = h("label", { class: "dropzone", for: "photo-file", tabindex: "0",
      onkeydown: (e) => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), fileInput.click()),
      ondragover: (e) => { e.preventDefault(); zone.classList.add("is-over"); },
      ondragleave: () => zone.classList.remove("is-over"),
      ondrop: (e) => { e.preventDefault(); zone.classList.remove("is-over"); e.dataTransfer.files[0] && choosePhoto(e.dataTransfer.files[0]); } },
      icon(ICONS.camera), h("strong", {}, "Take a photo or choose one"),
      h("span", { class: "hint" }, "Hold the phone straight above the page, in good light, with the whole answer in the picture."));
    return zone;
  }
  async function choosePhoto(file) {
    if (!file.type.startsWith("image/")) return toast("Please choose a photo (an image file).", "error");
    try {
      state.photoData = await shrinkPhoto(file, config.limits.photo_side || 2000);
    } catch {
      return toast("This photo couldn't be opened. Please try another photo.", "error");
    }
    mount(photoArea,
      h("div", { class: "photo-preview" }, h("img", { src: state.photoData, alt: "Your answer photo" }),
        h("button", { class: "btn btn--secondary btn--sm", type: "button", onclick: clearPhoto }, icon(ICONS.x), "Remove")),
      h("div", { style: { marginTop: "12px" } }, readBtn));
    toast("Photo ready. Press “Read my photo”.", "info");
  }
  function clearPhoto() {
    state.photoData = null;
    fileInput.value = "";
    mount(photoArea, dropzone());
  }
  async function readPhoto() {
    if (!state.photoData) return toast("Please take or choose a photo first.", "error");
    await withBusy(readBtn, "Reading your photo…", async () => {
      try {
        const res = await api("/photo", { method: "POST", body: { image: state.photoData }, timeout: 120000 });
        answerInput.value = res.text;
        store.setDraft({ answer: res.text, answerSource: "photo" });
        state.answerSource = "photo";
        unclearNote.hidden = !res.unclear;
        setAnswerTab("typed");
        answerInput.focus();
        toast(res.unclear ? "Answer read. Please fix the words marked [unclear]." : "Answer read from your photo. Please check it before checking.", res.unclear ? "warning" : "success");
      } catch (err) {
        toast(err.message, "error");
      }
    });
  }
  clearPhoto();

  // ----- Answer tabs -----
  const typePanel = h("div", { role: "tabpanel", id: "panel-type" }, h("div", { class: "field" }, answerInput, counter(answerInput, config.limits.answer)), unclearNote);
  const photoPanel = h("div", { role: "tabpanel", id: "panel-photo", hidden: true }, fileInput, photoArea,
    h("p", { class: "small muted", style: { margin: "10px 0 0" } }, "🔒 Your photo is only used to read your answer. It is never saved."));
  const tabType = h("button", { class: "tab", role: "tab", type: "button", "aria-controls": "panel-type", onclick: () => setAnswerTab("typed") }, icon(ICONS.pen), "Type");
  const tabPhoto = h("button", { class: "tab", role: "tab", type: "button", "aria-controls": "panel-photo", onclick: () => setAnswerTab("photo") }, icon(ICONS.camera), "Photo of written answer");
  [tabType, tabPhoto].forEach((t) => t.querySelector("svg").setAttribute("style", "width:18px;height:18px;fill:none;stroke:currentColor;stroke-width:2"));
  function setAnswerTab(which) {
    const photo = which === "photo";
    tabType.setAttribute("aria-selected", String(!photo));
    tabPhoto.setAttribute("aria-selected", String(photo));
    typePanel.hidden = photo;
    photoPanel.hidden = !photo;
  }
  setAnswerTab("typed");

  // ----- Result area -----
  const resultArea = h("div", { id: "result-area", tabindex: "-1" });
  showIntro();
  function showIntro() {
    mount(resultArea, h("div", { class: "card hero-card" },
      h("h2", {}, "How it works"),
      h("ol", { style: { margin: "0 0 4px", paddingLeft: "20px", display: "grid", gap: "6px" } },
        h("li", {}, "Pick your book and the type of question."),
        h("li", {}, "Type your question and answer, or photograph your notebook."),
        h("li", {}, "Get friendly feedback: what you did well, what to add, and which pages to reread.")),
      h("p", { style: { margin: "10px 0 0", fontSize: ".9rem" } },
        "Works for direct questions and HOTS questions that apply the book's ideas.")));
  }

  // ----- Check -----
  const checkBtn = h("button", { class: "btn btn--primary btn--lg btn--block", type: "button", onclick: runCheck }, icon(ICONS.check), "Check my answer");
  async function runCheck() {
    const name = student ? student.name : nameInput.value.trim();
    const problems = [];
    if (!student && !name) problems.push([nameInput, "Please type your name first."]);
    if (!state.bookId) problems.push([bookSelect, "Please choose a book first."]);
    if (!questionInput.value.trim()) problems.push([questionInput, "Please type your question first."]);
    if (!answerInput.value.trim()) problems.push([answerInput, "Your answer is empty. Please write your answer first."]);
    if (problems.length) {
      problems[0][0].focus();
      return toast(problems[0][1], "error");
    }
    const steps = loadingSteps();
    mount(resultArea, steps);
    resultArea.scrollIntoView({ behavior: "smooth", block: "start" });
    await withBusy(checkBtn, "Checking…", async () => {
      try {
        const res = await api("/check", { method: "POST", auth: "optional", body: {
          name, book_id: state.bookId, question_type: state.questionType, language: state.language,
          question: questionInput.value.trim(), answer: answerInput.value.trim(),
          answer_source: store.draft().answerSource === "photo" ? "photo" : "typed" } });
        steps.stop();
        showResult(res.check, res.review_proof);
        store.setDraft({ answerSource: "typed" });
        toast("Your answer has been checked. Your feedback is ready.", "success");
      } catch (err) {
        steps.stop();
        mount(resultArea, h("div", { class: "card" }, h("div", { class: "notice notice--danger" }, h("span", { class: "notice__icon" }, "⚠️"),
          h("div", {}, h("strong", {}, "Your answer couldn't be checked."), h("p", { style: { margin: "4px 0 0" } }, err.message),
            h("p", { class: "small muted", style: { margin: "4px 0 0" } }, "Your question and answer are still here."))),
          h("div", { class: "row", style: { marginTop: "12px" } }, h("button", { class: "btn btn--primary", type: "button", onclick: runCheck }, icon(ICONS.refresh), "Try again"))));
        toast(err.message, "error");
      }
    });
  }

  function showResult(check, proof) {
    const reviewArea = h("div");
    const reviewBtn = h("button", { class: "btn btn--secondary", type: "button", onclick: () => openReviewForm() }, icon(ICONS.flag), "Teacher, please review");
    function openReviewForm() {
      const reason = h("textarea", { class: "textarea", rows: "3", maxlength: String(config.limits.review_reason),
        placeholder: "Optional: why do you think the marking isn't fair?" });
      const send = h("button", { class: "btn btn--primary", type: "button" }, "Send to my teacher");
      send.addEventListener("click", () => withBusy(send, "Sending…", async () => {
        try {
          const res = await api("/review", { method: "POST", auth: "optional",
            body: { check_id: check.id, proof, reason: reason.value.trim() } });
          mount(reviewArea, h("div", { class: "notice notice--success" }, h("span", { class: "notice__icon" }, "✅"), h("div", {}, res.message)));
          reviewBtn.disabled = true;
          toast(res.already ? res.message : "Sent to your teacher for review.", "success");
        } catch (err) {
          toast(err.message, "error");
        }
      }));
      mount(reviewArea, h("div", { class: "card card--flat", style: { padding: "14px" } },
        h("p", { class: "small", style: { marginTop: 0 } }, h("strong", {}, "Think this was marked unfairly? "), "Your teacher has the final say."),
        h("div", { class: "field" }, reason), h("div", { class: "row" }, send,
          h("button", { class: "btn btn--ghost", type: "button", onclick: () => mount(reviewArea) }, "Cancel"))));
      reason.focus();
    }
    const actions = h("div", { class: "stack" },
      h("div", { class: "row" },
        h("button", { class: "btn btn--primary", type: "button", onclick: () => { answerInput.focus(); answerInput.scrollIntoView({ behavior: "smooth", block: "center" }); } },
          icon(ICONS.pen), "Improve my answer"),
        h("button", { class: "btn btn--secondary", type: "button", onclick: newQuestion }, "New question"),
        check.result?.covered_by_book ? reviewBtn : null),
      reviewArea);
    mount(resultArea, h("div", { class: "card" }, renderResult(check, { actions })));
    resultArea.focus({ preventScroll: true });
  }

  function newQuestion() {
    questionInput.value = "";
    answerInput.value = "";
    unclearNote.hidden = true;
    store.setDraft({ question: "", answer: "", answerSource: "typed" });
    showIntro();
    questionInput.focus();
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  // ----- Layout -----
  const noBooks = !books.length;
  const who = student
    ? h("div", { class: "notice notice--success" }, h("span", { class: "notice__icon" }, "👋"),
        h("div", {}, "Checking as ", h("strong", {}, student.name), ". Your results are saved to ", h("a", { href: "#/progress" }, "My Progress"), "."))
    : h("div", { class: "field" }, h("label", { class: "label", for: "name" }, "Your name"), nameInput,
        h("span", { class: "hint" }, "Want to track your progress? ", h("a", { href: "#/progress" }, "Create a free profile"), " (no email needed)."));

  mount(root,
    h("div", { class: "page-head" }, h("div", {},
      h("h1", {}, student ? `Hi ${student.name.split(" ")[0]}, let's practise! 👋` : "Practise. Check. Improve. ✨"),
      h("p", {}, "Write an answer to any question from your book and get instant, friendly feedback based on your own textbook."))),
    noBooks ? h("div", { class: "notice notice--warning", style: { marginBottom: "16px" } }, h("span", { class: "notice__icon" }, "📚"),
      h("div", {}, h("strong", {}, "No book yet. "), "Your teacher hasn't shared a book. Please check back soon.")) : null,
    h("div", { class: "layout-practice" },
      h("div", { class: "stack" },
        h("section", { class: "card", "aria-labelledby": "s1" },
          h("div", { class: "card__title" }, h("span", { class: "step" }, "1"), h("h2", { id: "s1" }, "Set up")),
          who,
          h("div", { class: "field" }, h("label", { class: "label", for: "book" }, "Book"), bookSelect),
          h("div", { class: "field" }, h("span", { class: "label", id: "qt-label" }, "Type of question"),
            segmented("qtype", (config.question_types || []).map((q) => ({ value: q.name, label: q.name, hint: `${q.key_points} key points` })),
              state.questionType, (v) => { state.questionType = v; store.setPrefs({ questionType: v }); })),
          h("div", { class: "field" }, h("span", { class: "label" }, "Feedback language"),
            segmented("lang", (config.languages || []).map((l) => ({ value: l, label: l })), state.language,
              (v) => { state.language = v; store.setPrefs({ language: v }); }))),
        h("section", { class: "card", "aria-labelledby": "s2" },
          h("div", { class: "card__title" }, h("span", { class: "step" }, "2"), h("h2", { id: "s2" }, "Your question")),
          h("div", { class: "field" }, h("label", { class: "sr-only", for: "question" }, "Your question"), questionInput, counter(questionInput, config.limits.question)),
          h("p", { class: "hint", style: { margin: 0 } }, "Direct questions and HOTS questions (case-based, 'what if', 'why') both work.")),
        h("section", { class: "card", "aria-labelledby": "s3" },
          h("div", { class: "card__title" }, h("span", { class: "step" }, "3"), h("h2", { id: "s3" }, "Your answer")),
          h("div", { class: "tabs", role: "tablist", "aria-label": "How to give your answer" }, tabType, tabPhoto),
          typePanel, photoPanel),
        h("div", { class: "sticky-cta" }, checkBtn)),
      h("div", { class: "stack" }, resultArea)));
  if (noBooks) checkBtn.disabled = true;
}
