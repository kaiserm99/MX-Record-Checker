"use client";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { Check, Send } from "lucide-react";
import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import { nominees, site } from "@/data/site";
import { Reveal, SectionHeading } from "./Reveal";

const categories = Array.from(new Set(nominees.map((n) => n.category))).concat(["Designated Driver", "Sonstiges"]);

const inputCls =
  "w-full rounded-xl border border-line bg-ink/60 px-4 py-3 text-foam placeholder:text-foam-3 transition-colors focus:border-gold/60 focus:bg-ink/80 focus:outline-none";

function Confetti() {
  const pieces = Array.from({ length: 26 }, (_, i) => ({
    x: (i / 26) * 100,
    r: (i * 47) % 360,
    d: 0.6 + ((i * 13) % 10) / 10,
    c: ["#f2b233", "#ffd97a", "#e0223f", "#7bd389", "#fff5dc"][i % 5],
  }));
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden" aria-hidden="true">
      {pieces.map((p, i) => (
        <motion.span
          key={i}
          className="absolute top-0 h-3 w-2 rounded-sm"
          style={{ left: `${p.x}%`, background: p.c }}
          initial={{ y: -20, rotate: 0, opacity: 1 }}
          animate={{ y: "110%", rotate: p.r + 360, opacity: [1, 1, 0] }}
          transition={{ duration: 1.8 * p.d, ease: "easeIn", delay: (i % 6) * 0.05 }}
        />
      ))}
    </div>
  );
}

export default function Nominate() {
  const id = useId();
  const reduced = useReducedMotion();
  const [sent, setSent] = useState(false);
  const cardRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // the card shrinks when the form disappears, keep the confirmation in view
    if (sent) cardRef.current?.scrollIntoView({ block: "center", behavior: reduced ? "auto" : "smooth" });
  }, [sent, reduced]);
  const [data, setData] = useState({ name: "", nick: "", category: categories[0], reason: "", by: "" });

  const onSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const subject = encodeURIComponent(`Nominierung ${site.year}: ${data.nick || data.name}`);
    const body = encodeURIComponent(
      [
        `Nominiert: ${data.name}`,
        `Spitzname: ${data.nick}`,
        `Kategorie: ${data.category}`,
        "",
        "Begründung:",
        data.reason,
        "",
        `Eingereicht von: ${data.by}`,
      ].join("\n"),
    );
    window.location.href = `mailto:${site.juryMail}?subject=${subject}&body=${body}`;
    setSent(true);
  };

  return (
    <section id="nominieren" className="section-pad relative scroll-mt-20">
      <div aria-hidden="true" className="pointer-events-none absolute inset-0 -z-10">
        <div className="absolute left-[10%] top-1/3 h-[40vh] w-[40vh] rounded-full bg-gold/10 blur-[110px]" />
        <div className="absolute right-[5%] bottom-0 h-[40vh] w-[40vh] rounded-full bg-cherry/10 blur-[110px]" />
      </div>

      <div className="container-x grid gap-12 lg:grid-cols-[1fr_1.1fr] lg:items-start">
        <div className="lg:sticky lg:top-32">
          <SectionHeading
            eyebrow="Nominieren"
            title={
              <>
                Du kennst jemanden? <span className="text-gold">Die Jury will Namen.</span>
              </>
            }
            text="Eine gute Nominierung ist eine gute Geschichte. Je konkreter, desto besser. Selbstnominierungen werden gelesen, gelacht und abgelehnt."
          />
          <Reveal delay={0.15} className="mt-8 rounded-2xl border border-line p-5 text-sm text-foam-2">
            <p className="font-semibold text-foam">Fair Play</p>
            <p className="mt-2 leading-relaxed">
              Nominiere nur Menschen, die davon wissen und mitlachen. Das ist eine Ehrenrunde unter Freund:innen,
              kein Pranger.
            </p>
          </Reveal>
        </div>

        <Reveal delay={0.1}>
          <div ref={cardRef} className="relative overflow-hidden rounded-3xl glass p-6 shadow-card sm:p-8">
            <AnimatePresence mode="wait">
              {sent ? (
                <motion.div
                  key="done"
                  initial={{ opacity: 0, scale: 0.96 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                  className="relative py-10 text-center"
                >
                  {!reduced && <Confetti />}
                  <motion.span
                    className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-gold text-ink shadow-glow"
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 260, damping: 16, delay: 0.1 }}
                  >
                    <Check size={30} aria-hidden="true" />
                  </motion.span>
                  <h3 className="mt-6 font-display text-2xl font-bold">Prost. Nominierung ist unterwegs.</h3>
                  <p className="mx-auto mt-3 max-w-sm text-sm text-foam-2">
                    Dein Mailprogramm sollte sich geöffnet haben. Falls nicht, schick die Geschichte einfach an{" "}
                    <a className="text-gold underline-offset-4 hover:underline" href={`mailto:${site.juryMail}`}>
                      {site.juryMail}
                    </a>
                    .
                  </p>
                  <button
                    type="button"
                    onClick={() => {
                      setSent(false);
                      setData({ name: "", nick: "", category: categories[0], reason: "", by: "" });
                    }}
                    className="mt-8 rounded-xl bg-foam/10 px-5 py-2.5 text-sm font-semibold hover:bg-foam/15"
                  >
                    Noch jemanden nominieren
                  </button>
                </motion.div>
              ) : (
                <motion.form
                  key="form"
                  onSubmit={onSubmit}
                  exit={{ opacity: 0, y: -8 }}
                  className="grid gap-5"
                >
                  <div className="grid gap-5 sm:grid-cols-2">
                    <div>
                      <label htmlFor={`${id}-name`} className="mb-1.5 block text-xs uppercase tracking-[0.2em] text-foam-3">
                        Name der Person
                      </label>
                      <input
                        id={`${id}-name`}
                        required
                        autoComplete="off"
                        className={inputCls}
                        placeholder="Wer soll es werden?"
                        value={data.name}
                        onChange={(e) => setData({ ...data, name: e.target.value })}
                      />
                    </div>
                    <div>
                      <label htmlFor={`${id}-nick`} className="mb-1.5 block text-xs uppercase tracking-[0.2em] text-foam-3">
                        Spitzname
                      </label>
                      <input
                        id={`${id}-nick`}
                        autoComplete="off"
                        className={inputCls}
                        placeholder="z. B. Bowle-Bernd"
                        value={data.nick}
                        onChange={(e) => setData({ ...data, nick: e.target.value })}
                      />
                    </div>
                  </div>

                  <div>
                    <label htmlFor={`${id}-cat`} className="mb-1.5 block text-xs uppercase tracking-[0.2em] text-foam-3">
                      Kategorie
                    </label>
                    <select
                      id={`${id}-cat`}
                      className={inputCls}
                      value={data.category}
                      onChange={(e) => setData({ ...data, category: e.target.value })}
                    >
                      {categories.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label htmlFor={`${id}-reason`} className="mb-1.5 block text-xs uppercase tracking-[0.2em] text-foam-3">
                      Die Geschichte
                    </label>
                    <textarea
                      id={`${id}-reason`}
                      required
                      minLength={20}
                      rows={5}
                      className={`${inputCls} resize-y`}
                      placeholder="Was ist passiert? Wo? Und warum reden alle noch davon?"
                      value={data.reason}
                      onChange={(e) => setData({ ...data, reason: e.target.value })}
                    />
                    <p className="mt-1.5 text-right text-xs text-foam-3">{data.reason.length} Zeichen</p>
                  </div>

                  <div>
                    <label htmlFor={`${id}-by`} className="mb-1.5 block text-xs uppercase tracking-[0.2em] text-foam-3">
                      Dein Name
                    </label>
                    <input
                      id={`${id}-by`}
                      required
                      autoComplete="name"
                      className={inputCls}
                      placeholder="Damit die Jury weiß, wem sie die Runde schuldet"
                      value={data.by}
                      onChange={(e) => setData({ ...data, by: e.target.value })}
                    />
                  </div>

                  <button
                    type="submit"
                    className="group mt-2 inline-flex items-center justify-center gap-2 rounded-2xl bg-gold px-6 py-3.5 font-semibold text-ink shadow-glow transition-transform hover:-translate-y-0.5 active:translate-y-0"
                  >
                    Nominierung abschicken
                    <Send size={18} className="transition-transform group-hover:translate-x-1 group-hover:-translate-y-0.5" aria-hidden="true" />
                  </button>
                  <p className="text-center text-xs text-foam-3">
                    Öffnet dein Mailprogramm. Es wird nichts auf dieser Seite gespeichert.
                  </p>
                </motion.form>
              )}
            </AnimatePresence>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
