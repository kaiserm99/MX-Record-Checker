export const site = {
  name: "Alkoholiker des Jahres",
  short: "ADJ",
  domain: "alkoholikerdesjahres.de",
  url: "https://alkoholikerdesjahres.de",
  year: 2026,
  tagline: "Die einzige Auszeichnung, bei der niemand nüchtern gewinnt.",
  description:
    "Alkoholiker des Jahres – die satirische Verleihung unter Freund:innen. Nominierte, Hall of Fame, Regeln und Countdown zur großen Gala 2026.",
  ceremonyDate: "2026-12-23T20:00:00+01:00",
  ceremonyLabel: "Mittwoch, 23.12.2026 · 20:00 Uhr",
  ceremonyPlace: "Stammtisch · Hinterzimmer · Eingang über den Hof",
  juryMail: "jury@alkoholikerdesjahres.de",
};

export type Nominee = {
  id: string;
  nick: string;
  title: string;
  category: string;
  drink: string;
  quote: string;
  hue: number;
  initials: string;
};

export const nominees: Nominee[] = [
  {
    id: "kapitaen",
    nick: "Der Kapitän",
    title: "Hält das Schiff auf Kurs, egal wie hoch der Wellengang.",
    category: "Standhaftigkeit",
    drink: "Dunkles vom Fass, immer im Krug",
    quote: "Ein Bier ist kein Bier. Ein Krug ist ein Argument.",
    hue: 38,
    initials: "KP",
  },
  {
    id: "frieda",
    nick: "Frühschoppen-Frieda",
    title: "Erste am Tresen, letzte beim Aufräumen.",
    category: "Frühdienst",
    drink: "Weißbier mit Weißwurst, 10:00 Uhr scharf",
    quote: "Wer früh anfängt, hat den Sonntag länger.",
    hue: 12,
    initials: "FF",
  },
  {
    id: "tobi",
    nick: "Tequila-Tobi",
    title: "Der Mann, der Salz und Zitrone für Beilagen hält.",
    category: "Kreativität",
    drink: "Alles, was in ein Shotglas passt",
    quote: "Ich mische nicht. Ich kombiniere.",
    hue: 330,
    initials: "TT",
  },
  {
    id: "rainer",
    nick: "Radler-Rainer",
    title: "Trinkt seit 2014 »nur eins«. Zählt aber nie mit.",
    category: "Ausdauer",
    drink: "Radler. Angeblich.",
    quote: "Das ist doch fast ein Saft.",
    hue: 88,
    initials: "RR",
  },
  {
    id: "gin",
    nick: "Gin-Gisela",
    title: "Kennt 47 Tonics. Trinkt trotzdem immer das gleiche.",
    category: "Stil",
    drink: "Gin Tonic, Gurke, Eis aus dem Silikonwürfel",
    quote: "Klasse statt Masse. Und dann doch Masse.",
    hue: 195,
    initials: "GG",
  },
  {
    id: "spaeti",
    nick: "Späti-Sascha",
    title: "Weiß, welcher Kiosk um 4 Uhr noch Eis hat.",
    category: "Logistik",
    drink: "Sternburg, Tüte, Bordstein",
    quote: "Der Abend ist erst vorbei, wenn die Bäckerei aufmacht.",
    hue: 268,
    initials: "SS",
  },
];

export type Winner = {
  year: number;
  nick: string;
  reason: string;
  highlight?: boolean;
};

export const winners: Winner[] = [
  { year: 2025, nick: "Bowle-Bernd", reason: "Hat auf einer Hochzeit die Bowle nachgefüllt. Aus dem Kofferraum.", highlight: true },
  { year: 2024, nick: "Aperol-Anke", reason: "Drei Sommer, ein Getränk, null Reue." },
  { year: 2023, nick: "Der Kapitän", reason: "Erster Doppelsieg in der Geschichte der Verleihung." },
  { year: 2022, nick: "Der Kapitän", reason: "Hielt eine 40-minütige Rede über den perfekten Schaum." },
  { year: 2021, nick: "Zoom-Zoe", reason: "Feierabendbier vor laufender Kamera. Jeden Tag. Ein Jahr lang." },
  { year: 2020, nick: "Balkon-Basti", reason: "Hat den Balkon zum Biergarten erklärt. Inklusive Bedienung." },
  { year: 2019, nick: "Kellerbar-Karl", reason: "Gründungsjahr. Gründungsmitglied. Gründungsrausch." },
];

export const stats = [
  { value: 8, suffix: "", label: "Jahre Tradition" },
  { value: 87, suffix: "", label: "Nominierte bisher" },
  { value: 0, suffix: "", label: "Nüchterne Jurymitglieder" },
  { value: 1, suffix: " Pokal", label: "Pro Jahr. Immer noch derselbe." },
];

export const rules = [
  {
    step: "01",
    title: "Nominieren",
    text: "Jede:r kann nominieren, sich selbst ausgenommen. Eine gute Geschichte zählt mehr als eine hohe Zahl.",
  },
  {
    step: "02",
    title: "Jury tagt",
    text: "Fünf Jurymitglieder, ein Hinterzimmer, eine Nacht. Entscheidungen sind endgültig und selten nachvollziehbar.",
  },
  {
    step: "03",
    title: "Gala",
    text: "Am 23. Dezember wird verliehen, jedes Jahr. Roter Teppich optional, Dankesrede Pflicht, Länge nach oben offen.",
  },
  {
    step: "04",
    title: "Pokal wandert",
    text: "Der Pokal bleibt ein Jahr. Wer ihn verliert, zahlt die nächste Runde. Für alle. Für immer.",
  },
];

export const faq = [
  {
    q: "Ist das ernst gemeint?",
    a: "Nein. Das ist eine Auszeichnung unter Freund:innen, die sich über ihre eigenen Stammtisch-Legenden lustig machen. Niemand hier findet Sucht witzig. Wer merkt, dass es kein Spaß mehr ist, findet unten im Footer echte Hilfe.",
  },
  {
    q: "Kann ich jemanden nominieren, der nicht am Stammtisch ist?",
    a: "Nur, wenn die Person davon weiß und lacht. Wir nominieren keine Menschen gegen ihren Willen. Das ist keine Pranger-Seite, sondern eine Ehrenrunde.",
  },
  {
    q: "Was gewinnt man?",
    a: "Den Wanderpokal, ein Jahr Ruhm im Freundeskreis und das Recht, die Playlist auf der nächsten Party zu bestimmen. Das ist mehr, als es klingt.",
  },
  {
    q: "Wer sitzt in der Jury?",
    a: "Die Gewinner:innen der letzten fünf Jahre. Befangenheit ist Teil des Konzepts.",
  },
  {
    q: "Gibt es auch alkoholfreie Kategorien?",
    a: "Ja. »Designated Driver des Jahres« ist seit 2022 die am härtesten umkämpfte Kategorie. Der Preis ist ein voller Tank.",
  },
];

export const marquee = [
  "Prost",
  "Cheers",
  "Salud",
  "Santé",
  "Skål",
  "Na zdrowie",
  "Kanpai",
  "Cin cin",
  "Proost",
  "Slàinte",
  "Şerefe",
  "Saúde",
];
