/**
 * Builds the site as a static export and folds it into ONE self-contained HTML file
 * (out/single.html): every script, stylesheet and font is inlined, so the page can be
 * hosted anywhere a single HTML document can be served (no /_next/ paths, no runtime fetches).
 *
 * Usage: node scripts/build-single-file.mjs
 */
import { execSync } from "node:child_process";
import { readFileSync, writeFileSync, mkdirSync } from "node:fs";
import { join, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const hero3d = join(root, "src/components/Hero3D.tsx");
const original = readFileSync(hero3d, "utf8");

// The scene is normally a lazily fetched chunk; in a single file there is nothing to fetch,
// so the build temporarily imports it statically.
const staticHero3D = `"use client";
import Scene from "./Scene";
export default function Hero3D() {
  return (
    <div className="absolute inset-0" aria-hidden="true">
      <Scene />
    </div>
  );
}
`;

try {
  writeFileSync(hero3d, staticHero3D);
  execSync("npx next build --webpack", { cwd: root, stdio: "inherit", env: { ...process.env, STATIC_EXPORT: "1" } });
} finally {
  writeFileSync(hero3d, original);
}

const out = join(root, "out");
const mime = { ".woff2": "font/woff2", ".woff": "font/woff", ".ttf": "font/ttf", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon" };
const readOut = (p) => readFileSync(join(out, p.replace(/^\//, "").split("?")[0]));
const dataUri = (p) => `data:${mime[extname(p.split("?")[0])] ?? "application/octet-stream"};base64,${readOut(p).toString("base64")}`;

let html = readFileSync(join(out, "index.html"), "utf8");

// stylesheets → <style> with fonts inlined
html = html.replace(/<link([^>]*?)href="([^"]+\.css[^"]*)"([^>]*)>/g, (m, a, href) => {
  if (!/rel="stylesheet"/.test(a + m)) return "";
  let css = readOut(href).toString("utf8");
  const cssDir = dirname(href.split("?")[0]);
  css = css.replace(/url\((["']?)([^)"']+)\1\)/g, (m, q, p) => {
    if (/^(data:|https?:|#)/.test(p) || p.startsWith("%23")) return m;
    const abs = p.startsWith("/") ? p : join(cssDir, p);
    return `url(${dataUri(abs)})`;
  });
  return `<style>${css}</style>`;
});
// preload / prefetch hints are pointless once everything is inline
html = html.replace(/<link[^>]*rel="(preload|modulepreload|prefetch)"[^>]*>/g, "");
// icon → data uri
html = html.replace(/<link([^>]*?)href="(\/icon\.svg[^"]*)"/g, (m, a, href) => `<link${a}href="${dataUri(href)}"`);
// external scripts → inline, in document order
html = html.replace(/<script([^>]*?)src="(\/_next\/[^"]+\.js[^"]*)"([^>]*)><\/script>/g, (m, a, src, b) => {
  const attrs = (a + b).replace(/\s(async|defer|crossorigin(="[^"]*")?)/g, "");
  let js = readOut(src).toString("utf8").replace(/<\/script/gi, "<\\/script");
  // Next derives its asset prefix from document.currentScript.src, which an inline script lacks
  // some libraries carry a literal U+FFFD in string/regex literals; the escape is equivalent and survives any host
  js = js.replace(/\uFFFD/g, "\\uFFFD");
  js = js.replace(/new URL\((\w+)\.src\)(,\w+=\w+\.indexOf\("\/_next\/"\))/g, 'new URL($1.src||"/_next/",location.href)$2');
  return `<script${attrs}>${js}</script>`;
});

// The RSC payload still asks React to hoist the layout stylesheet, font preloads and the icon
// by URL; neutralise those so nothing is fetched (everything is already inline).
html = html.replace(/\\"rel\\":\\"(stylesheet|preload)\\",\\"href\\":\\"\/_next\//g, '\\"rel\\":\\"x-inlined\\",\\"href\\":\\"/_next/');
html = html.replace(/\\"rel\\":\\"icon\\",\\"href\\":\\"\/icon\.svg[^\\]*\\"/g, () => `\\"rel\\":\\"icon\\",\\"href\\":\\"${dataUri("/icon.svg")}\\"`);

// ...and drop the payload's resource hints (":HL" rows) for the same assets
html = html.replace(/:HL\[\\"\/_next\/.*?\]\\n/g, "");

if (/\/_next\//.test(html)) {
  const left = [...html.matchAll(/\/_next\/[^"'\s)]+/g)].map((m) => m[0]);
  console.warn("references left to /_next/:", [...new Set(left)].slice(0, 10));
}

mkdirSync(out, { recursive: true });
writeFileSync(join(out, "single.html"), html);
console.log(`out/single.html written (${(html.length / 1024 / 1024).toFixed(2)} MB)`);

// Fragment variant for hosts that wrap the page in their own <html>/<head>/<body>
// (e.g. claude.ai artifacts): head + body contents only, root attributes re-applied by script.
const htmlAttrs = html.match(/<html([^>]*)>/)?.[1] ?? "";
const bodyAttrs = html.match(/<body([^>]*)>/)?.[1] ?? "";
const head = html.match(/<head>([\s\S]*?)<\/head>/)?.[1] ?? "";
const body = html.match(/<body[^>]*>([\s\S]*)<\/body>/)?.[1] ?? "";
const title = head.match(/<title>[^<]*<\/title>/)?.[0] ?? "";
const attr = (src, name) => src.match(new RegExp(`${name}="([^"]*)"`))?.[1] ?? "";
// next/font sets its CSS variables through classes on <html>; mirror them on :root too
const fontVars = [...html.matchAll(/\.[\w-]+__variable\{([^}]*)\}/g)].map((m) => m[1]).join(";");
const boot = `<script>(function(){var h=document.documentElement;h.lang=${JSON.stringify(attr(htmlAttrs, "lang"))};h.className+=" "+${JSON.stringify(attr(htmlAttrs, "class"))};document.body.className+=" "+${JSON.stringify(attr(bodyAttrs, "class"))};})()</script>`;
const fragment = `${title}\n<style>:root{${fontVars};color-scheme:dark;padding:0}body{margin:0}</style>\n${head.replace(title, "")}\n${boot}\n${body}`;
writeFileSync(join(out, "artifact.html"), fragment);
console.log(`out/artifact.html written (${(fragment.length / 1024 / 1024).toFixed(2)} MB)`);
