// A small, dependency-free line chart for the progress page.
import { h } from "./ui.js";

export function lineChart(points) {
  const W = 640, H = 240, pad = { l: 36, r: 16, t: 16, b: 30 };
  const innerW = W - pad.l - pad.r, innerH = H - pad.t - pad.b;
  const n = points.length;
  const x = (i) => pad.l + (n === 1 ? innerW / 2 : (i * innerW) / (n - 1));
  const y = (v) => pad.t + innerH - (v / 100) * innerH;
  const tip = h("div", { class: "chart-tip", hidden: true });
  const coords = points.map((p, i) => [x(i), y(p.score)]);
  const linePath = coords.map(([cx, cy], i) => `${i ? "L" : "M"}${cx.toFixed(1)},${cy.toFixed(1)}`).join(" ");
  const areaPath = `${linePath} L${x(n - 1).toFixed(1)},${y(0)} L${x(0).toFixed(1)},${y(0)} Z`;
  const labelEvery = Math.max(1, Math.ceil(n / 10));

  const showTip = (p, cx, cy, svg) => {
    const box = svg.getBoundingClientRect();
    tip.textContent = `Check ${p.n}: ${p.score}% · ${p.question}`;
    tip.style.left = `${(cx / W) * box.width}px`;
    tip.style.top = `${(cy / H) * box.height}px`;
    tip.hidden = false;
  };

  const svg = h("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": `Line chart of ${n} scores` },
    h("defs", {}, h("linearGradient", { id: "chartFill", x1: "0", y1: "0", x2: "0", y2: "1" },
      h("stop", { offset: "0", "stop-color": "var(--primary)", "stop-opacity": "0.25" }),
      h("stop", { offset: "1", "stop-color": "var(--primary)", "stop-opacity": "0" }))),
    [0, 25, 50, 75, 100].map((v) => h("g", {},
      h("line", { class: "grid-line", x1: String(pad.l), x2: String(W - pad.r), y1: String(y(v)), y2: String(y(v)) }),
      h("text", { class: "axis-label", x: String(pad.l - 8), y: String(y(v) + 4), "text-anchor": "end" }, String(v)))),
    points.map((p, i) => (i % labelEvery === 0 || i === n - 1)
      ? h("text", { class: "axis-label", x: String(x(i)), y: String(H - 8), "text-anchor": "middle" }, String(p.n)) : null),
    n > 1 ? h("path", { class: "area", d: areaPath }) : null,
    n > 1 ? h("path", { class: "line", d: linePath }) : null);
  points.forEach((p, i) => {
    const [cx, cy] = coords[i];
    const dot = h("circle", { class: "point", cx: String(cx), cy: String(cy), r: "5", tabindex: "0",
      "aria-label": `Check ${p.n}: ${p.score} percent` });
    dot.addEventListener("mouseenter", () => showTip(p, cx, cy, svg));
    dot.addEventListener("focus", () => showTip(p, cx, cy, svg));
    dot.addEventListener("click", () => showTip(p, cx, cy, svg));
    dot.addEventListener("mouseleave", () => { tip.hidden = true; });
    dot.addEventListener("blur", () => { tip.hidden = true; });
    svg.append(dot);
  });
  return h("div", { class: "chart" }, svg, tip,
    h("p", { class: "small muted", style: { margin: "6px 0 0", textAlign: "center" } }, "Check number (oldest to newest). Tap a point for details."));
}
