# alkoholikerdesjahres.de

Satirische One-Page-Website für die Verleihung „Alkoholiker des Jahres“ unter Freund:innen.
Gebaut mit Next.js 16 (App Router), Tailwind CSS 4, React Three Fiber und Framer Motion.

## Features

- 3D-Hero mit goldenem Wanderpokal, Kronkorken, Bläschen und Funkeln (Three.js via `@react-three/fiber` + `@react-three/drei`), Maus-Parallax und Scroll-Parallax
- Live-Countdown zur Gala (Rolling Digits), animierte Statistiken
- Nominierten-Karten mit 3D-Tilt und Glow, „Daumen drücken“ wird lokal im Browser gespeichert
- Hall of Fame mit scroll-gesteuerter Timeline
- Nominierungsformular (öffnet das Mailprogramm, kein Backend nötig)
- FAQ-Accordion, Marquee, Glas-Navigation mit Scroll-Progress, mobiles Menü
- SEO: Metadata, Open-Graph-Bild (generiert), `robots.txt`, `sitemap.xml`, SVG-Icon
- Respektiert `prefers-reduced-motion`, Fokus-Styles und ARIA-Attribute
- Footer mit Hinweis auf die Sucht & Drogen Hotline der BZgA

## Entwicklung

```bash
npm install
npm run dev
```

Produktion:

```bash
npm run build
npm start
```

## Inhalte anpassen

Alle Texte, Nominierten, Gewinner:innen, Regeln, FAQ und der Termin liegen in `src/data/site.ts`.
Farben, Schriften und Animationen sind als Tailwind-Tokens in `src/app/globals.css` definiert.

## Struktur

```
src/app/            Layout, Seite, globales CSS, OG-Bild, robots, sitemap, Icon
src/components/     Sektionen (Hero, Countdown, Nominees, HallOfFame, Rules, Nominate, FAQ, Footer)
src/components/Scene.tsx   Die 3D-Szene (wird nur im Browser geladen)
src/data/site.ts    Inhalte
```

Die Nominierten sind frei erfunden. Ab 18. Trink verantwortungsvoll.
