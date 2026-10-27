# AnswerMitra – website

**Your study friend that checks your answers against your own textbook.**

Students practise answers at home but usually get no feedback until a test is marked weeks
later, and a teacher with 40 students can't check every practice answer. AnswerMitra gives
instant, encouraging feedback based on the student's own prescribed book, and keeps the
teacher in control.

This folder is the website (HTML, CSS and JavaScript). It is published free with
**GitHub Pages** and talks to your **Hugging Face Space**, which does the AI checking and
keeps all the secret keys safe.

```
Students' phones / classroom laptops
            │
            ▼
GitHub Pages website  (this folder: HTML + CSS + JavaScript, no secrets)
            │  HTTPS, JSON
            ▼
Hugging Face Space  (app.py: Gemini AI, teacher password, saving)
            │
            ▼
Private Hugging Face Dataset  (books, profiles, results)
```

## Features

**Students (Practice)**
- Choose a book, **Short** or **Long** answer, and feedback in **English**, **Simple English** or **हिंदी (Hindi)**.
- Type an answer or **photograph a handwritten answer**; the photo is shrunk on the phone, read by
  Gemini, and never saved. Words that couldn't be read are marked `[unclear]` for the student to fix.
- Friendly results: score ring, "You covered 3 of 5 key points", **What you did well**,
  **Add these to improve**, statements to check, tips, a model answer and the pages to reread.
- Works for direct and **HOTS** (application) questions.
- **Teacher, please review** button for checks a student thinks were marked unfairly.
- Drafts are saved on the device, so nothing is lost if the page is closed.

**Students (My Progress, optional profile)**
- A profile needs only a name and a password, no email. Students stay signed in on their own device.
- Stats, an improvement trend, a score chart, averages by question type / kind / book,
  strong topics and topics to practise, and every past check with the teacher's reply.

**Teachers**
- Password-protected dashboard (signed in only until the tab is closed).
- Review requests with a marking panel, a results table with search and CSV download,
  drag-and-drop upload of several PDF books with show/hide switches, and password resets.
- "AI vs teacher" agreement score: how often the AI's mark was within 10 points of yours.

**Design**
- Mobile-first, with a bottom tab bar on phones, light and dark mode, large touch targets,
  keyboard and screen-reader friendly, clear loading messages and pop-up confirmations for
  every action. Can be added to a phone's home screen like an app.

## Publish it (about 15 minutes)

### Step 1 – Update your Hugging Face Space

The website needs the newest `app.py` (it adds the website's API).

1. Open your Space, click **Files → + Add file → Upload files**.
2. Upload the new **app.py** from the `Project Data` folder and click **Commit changes to main**.
3. Wait until the Space shows **Running**.
4. Check it: open `https://YOUR-SPACE-ADDRESS/api/v1/health` in a browser.
   You should see `{"ok":true,...}`.

Your Space address is `https://OWNER-SPACENAME.hf.space`. For
`https://huggingface.co/spaces/hiteshgoyal2009/class-quiz` it is
`https://hiteshgoyal2009-class-quiz.hf.space`.

### Step 2 – Point the website at your Space

Open `js/config.js` and check `apiBase` is your Space address (it is already set to
`https://hiteshgoyal2009-class-quiz.hf.space`).

### Step 3 – Put the website on GitHub

1. Sign in at **https://github.com** (create a free account if needed).
2. Click **+ → New repository**. Name it, for example, `answermitra`. Choose **Public**
   (GitHub Pages is free for public repositories). Click **Create repository**.
3. On the new repository page, click **uploading an existing file**.
4. Drag **everything inside this `Web App` folder** (not the folder itself) into the page:
   `index.html`, `manifest.webmanifest`, `.nojekyll`, `README.md`, and the `css`, `js` and
   `assets` folders. Click **Commit changes**.
   - On a Mac, press `Cmd + Shift + .` in Finder to see the hidden `.nojekyll` file.

### Step 4 – Turn on GitHub Pages

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **Deploy from a branch**, branch **main**, folder **/ (root)**, then **Save**.
3. After a minute or two, the page shows your website address, like
   `https://YOUR-GITHUB-NAME.github.io/answermitra/`.

### Step 5 (recommended) – Allow only your website

By default any website may call your Space's API (safe, because sign-in uses tokens, but
tidier to restrict). In your Space, open **Settings → Variables and secrets → New variable**:

| Name | Value |
|---|---|
| `ALLOWED_ORIGINS` | `https://YOUR-GITHUB-NAME.github.io` |

Optional: add a secret `APP_SECRET` with a long random sentence. It signs sign-in tokens;
changing it signs everyone out.

## Using it in a live classroom

1. **5–10 minutes before class**, open the website yourself. If the Space was asleep you'll
   see "Waking up AnswerMitra…"; wait until the Practice page appears.
2. Log in on the **Teacher** tab and check the right books are switched on.
3. Share the website link with students (write it on the board, or show a QR code: in
   Chrome, open the page and choose **Share → QR code**).
4. Students can **Add to Home Screen** on their phones to open it like an app.
5. During class, keep the **Teacher → Reviews** tab open to answer review requests.
6. After class, open **Results** and press **Download CSV**.

**Gemini quota:** each check uses two Gemini requests, and reading a photo uses one more.
Check your remaining free quota in Google AI Studio before a busy lesson or a presentation.

## Try it on your computer first (optional)

The website must be served (not opened as a file). In a terminal, inside this folder:

```bash
python3 -m http.server 8000
```

Then open `http://localhost:8000`. It uses the Space address from `js/config.js`.

## Folder structure

```
index.html              page shell: top bar, navigation, toasts
manifest.webmanifest    lets phones add it to the home screen
.nojekyll               tells GitHub Pages to serve files as they are
css/styles.css          design system, light/dark themes, responsive layout
js/config.js            your Space address (the only file you need to edit)
js/app.js               starts the app, wakes the server, switches pages
js/api.js               talks to the Space API (timeouts, errors, uploads, downloads)
js/store.js             remembers sign-ins, drafts and choices on the device
js/ui.js                safe element building, toasts, dialogs, helpers
js/result.js            the feedback display (score ring, sections)
js/charts.js            the progress chart (no libraries)
js/views/practice.js    Practice page
js/views/progress.js    My Progress page
js/views/teacher.js     Teacher dashboard
assets/favicon.svg      logo / app icon
```

## Troubleshooting

| What you see | What to do |
|---|---|
| "Waking up AnswerMitra…" for a long time | Open the Space on Hugging Face to check it says **Running**. Free Spaces sleep after a while and take a minute to start. |
| "Can't reach the AnswerMitra server" | Check `apiBase` in `js/config.js`, and that `https://YOUR-SPACE-ADDRESS/api/v1/health` works. If you set `ALLOWED_ORIGINS`, it must exactly match your website address (no trailing slash, no `/answermitra`). |
| Changes don't appear on the website | GitHub Pages can take 1–2 minutes to update. Then refresh with `Ctrl + F5` (Windows) or `Cmd + Shift + R` (Mac). |
| "No book yet" | Upload a book on the Teacher tab and make sure its switch is on. |
| Students are signed out | The teacher reset their password, or `APP_SECRET` changed. They just sign in again. |

## Privacy

- No email or phone number is collected.
- Passwords are never stored, only salted PBKDF2 hashes.
- Photos are used only to read the answer and are never saved.
- All books, profiles and results stay in the teacher's private Hugging Face Dataset.
- Questions, answers and photos are sent to Google Gemini for checking.
- The website itself stores only the student's sign-in, choices and draft on their own device.
