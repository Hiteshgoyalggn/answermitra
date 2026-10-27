---
title: AnswerMitra
emoji: 📝
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
---

# AnswerMitra

**Your study friend that checks your answers against your own textbook.**

Students practise answers at home but usually get no feedback until a test is marked weeks
later, and a teacher with 40 students can't check every practice answer. AnswerMitra gives
instant, encouraging feedback based on the student's own prescribed book, and keeps the
teacher in control.

## For students (Check My Answer tab)

- Choose a book, **Short answer** or **Long answer**, and a feedback language:
  **English**, **Simple English** or **हिंदी (Hindi)**.
- Type a question and an answer, or **photograph a handwritten answer** from a notebook.
  Gemini reads the handwriting into the answer box, the student fixes any misread words,
  then checks it. The photo itself is never saved.
- Feedback starts with what went well: "You covered 3 of 5 key points", then
  **What you did well**, **Add these to improve**, statements to check against the book,
  tips, a model answer based on the book, and the **PDF pages to reread**.
- Works for **direct** questions and **indirect / HOTS** questions (applying the book's ideas
  to a new situation). Before marking, Gemini reads an outline of the book and picks the
  concepts and pages the question needs.
- **Teacher, please review:** if a student thinks a check was marked unfairly, they can send
  it to the teacher with a reason. The teacher has the final say.

## For students (My Profile tab, optional)

- A profile needs only a name and a password, no email.
- Shows a progress report (average, best, latest, improving or not, averages by question type,
  direct/HOTS and book, strong topics and topics to practise), a chart of scores, and previous
  checks with their full feedback and any teacher review.

## For teachers (Teacher tab, password protected)

- Upload several books at once and choose which ones students can see.
- See every check (typed or photo, feedback language, AI mark, teacher mark), filter by book,
  and download a CSV.
- **Review requests and your marks:** answer students' review requests with your own mark and
  a comment. You can also mark any check yourself, and the app reports how often the AI's mark
  was within 10 points of yours.
- Set a new password for a student who forgot theirs.

## Privacy by design

- No email or phone number is collected.
- Passwords are never stored, only salted PBKDF2 hashes.
- Photos are used only to read the answer and are never saved.
- All books, profiles and results are kept in a private Hugging Face Dataset that only the
  teacher can open. Students' questions, answers and photos are sent to Google Gemini for checking.

## Website (GitHub Pages)

The `Web App` folder is a modern HTML/CSS/JavaScript website for classroom use. It calls this
Space through a JSON API under `/api/v1/` (see section 8 of `app.py`), so no secrets are in the
website. Optional Space settings: `ALLOWED_ORIGINS` (your website address, e.g.
`https://name.github.io`) and `APP_SECRET` (signs sign-in tokens). See `Web App/README.md`.

## Speed

Everything is loaded into memory when the Space starts, saves are copied to the Dataset in the
background (several at a time), and Gemini is asked to keep its thinking short. Students get their
result as soon as Gemini replies, without waiting for uploads.

## AI requests per check

Each check uses two Gemini requests (one to find the right pages, one to mark). Reading a photo
uses one more. To use one request per check, set `SMART_SEARCH = False` near the top of
`app.py`; HOTS questions are then matched to the book less reliably.

## Required secrets (Space → Settings → Variables and secrets)

| Name | What it is |
|---|---|
| `GEMINI_API_KEY` | Your key from Google AI Studio |
| `HF_TOKEN` | A Hugging Face token with **write** access |
| `DATASET_REPO` | Your private dataset, e.g. `your-username/classroom-data` |
| `TEACHER_PASSWORD` | The password for the Teacher tab |

Optional variable: `GEMINI_MODEL` (default `gemini-3.5-flash-lite`). Change it if Google
renames or retires the model.

## What is stored in the Dataset repo

```
books/<book>/meta.json        book title, page count, and whether students can see it
books/<book>/pages.json       the text of every page
self_checks/<id>.json         one file per student check, its result, and any teacher review
profiles/<name>.json          a student profile: name and salted password hash (no email)
```
