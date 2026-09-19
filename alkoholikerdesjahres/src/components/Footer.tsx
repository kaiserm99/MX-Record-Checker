import { HeartHandshake, Phone } from "lucide-react";
import { site } from "@/data/site";

export default function Footer() {
  return (
    <footer className="relative border-t border-line">
      <div className="container-x py-14">
        <div className="grid gap-10 md:grid-cols-[1.4fr_1fr_1fr]">
          <div>
            <p className="font-display text-2xl font-extrabold">
              {site.name} <span className="text-gold">{site.year}</span>
            </p>
            <p className="mt-3 max-w-md text-sm leading-relaxed text-foam-2">
              Eine satirische Auszeichnung unter Freund:innen. Wir feiern Geschichten, nicht Promille. Ab 18. Trink
              verantwortungsvoll und lass das Auto stehen.
            </p>
          </div>

          <nav aria-label="Footer">
            <p className="text-xs uppercase tracking-[0.2em] text-foam-3">Seite</p>
            <ul className="mt-3 space-y-2 text-sm">
              {[
                ["#nominierte", "Nominierte"],
                ["#hall-of-fame", "Hall of Fame"],
                ["#regeln", "Regeln"],
                ["#nominieren", "Nominieren"],
                ["#faq", "FAQ"],
              ].map(([href, label]) => (
                <li key={href}>
                  <a href={href} className="text-foam-2 transition-colors hover:text-gold">
                    {label}
                  </a>
                </li>
              ))}
            </ul>
          </nav>

          <div className="rounded-2xl border border-gold/30 bg-gold/5 p-5">
            <p className="flex items-center gap-2 text-sm font-semibold text-gold">
              <HeartHandshake size={18} aria-hidden="true" />
              Wenn es kein Spaß mehr ist
            </p>
            <p className="mt-2 text-sm leading-relaxed text-foam-2">
              Sucht &amp; Drogen Hotline der BZgA, rund um die Uhr, anonym:
            </p>
            <a
              href="tel:+4918063130310"
              className="mt-2 inline-flex items-center gap-2 font-display text-lg font-bold text-foam hover:text-gold"
            >
              <Phone size={18} aria-hidden="true" />
              01806 31 30 31
            </a>
            <p className="mt-1 text-xs text-foam-3">20 ct/Anruf aus dem Festnetz, max. 60 ct aus dem Mobilfunk.</p>
            <a
              href="https://www.kenn-dein-limit.de"
              target="_blank"
              rel="noopener noreferrer"
              className="mt-3 inline-block text-sm text-gold underline-offset-4 hover:underline"
            >
              kenn-dein-limit.de
            </a>
          </div>
        </div>

        <div className="mt-12 flex flex-col gap-3 border-t border-line pt-6 text-xs text-foam-3 sm:flex-row sm:items-center sm:justify-between">
          <p>
            © {new Date().getFullYear()} {site.domain} · Alle Nominierten sind frei erfunden. Ähnlichkeiten mit deinem
            Stammtisch sind beabsichtigt.
          </p>
          <p>Mit Schaum gebaut. Kein Cookie, kein Tracking, kein Kater.</p>
        </div>
      </div>
    </footer>
  );
}
