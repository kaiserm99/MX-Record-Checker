"use client";

import { AnimatePresence, motion } from "framer-motion";
import { CalendarDays, MapPin } from "lucide-react";
import { useSyncExternalStore } from "react";
import { Reveal, SectionHeading } from "./Reveal";
import { site } from "@/data/site";

type Parts = { d: number; h: number; m: number; s: number } | null;

function diff(target: number): Parts {
  const ms = target - Date.now();
  if (ms <= 0) return { d: 0, h: 0, m: 0, s: 0 };
  return {
    d: Math.floor(ms / 86_400_000),
    h: Math.floor((ms / 3_600_000) % 24),
    m: Math.floor((ms / 60_000) % 60),
    s: Math.floor((ms / 1000) % 60),
  };
}

function Digit({ value }: { value: string }) {
  return (
    <span className="relative inline-block h-[1em] w-[0.6em] overflow-hidden">
      <AnimatePresence initial={false}>
        <motion.span
          key={value}
          className="absolute inset-0 grid place-items-center"
          initial={{ y: "100%", opacity: 0 }}
          animate={{ y: 0, opacity: 1 }}
          exit={{ y: "-100%", opacity: 0 }}
          transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
        >
          {value}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

function Unit({ value, label }: { value: number | null; label: string }) {
  const str = value === null ? "--" : String(value).padStart(2, "0");
  return (
    <div className="glass flex flex-col items-center rounded-2xl px-1.5 py-5 sm:px-3 sm:py-7">
      <div className="flex justify-center whitespace-nowrap font-display text-3xl font-extrabold tabular-nums leading-none text-foam sm:text-5xl xl:text-6xl">
        {str.split("").map((c, i) => (
          <Digit key={i} value={c} />
        ))}
      </div>
      <span className="mt-3 text-[11px] uppercase tracking-[0.25em] text-foam-3">{label}</span>
    </div>
  );
}

const target = new Date(site.ceremonyDate).getTime();

function subscribe(onTick: () => void) {
  const id = setInterval(onTick, 1000);
  return () => clearInterval(id);
}
const getNow = () => Math.floor(Date.now() / 1000);
const getServerNow = () => null;

export default function Countdown() {
  // null on the server and during hydration, so markup matches and the digits roll in on the client
  const now = useSyncExternalStore(subscribe, getNow, getServerNow);
  const parts: Parts = now === null ? null : diff(target);

  return (
    <section id="countdown" className="section-pad relative">
      <div className="container-x grid gap-12 lg:grid-cols-[1fr_1.2fr] lg:items-center">
        <div>
          <SectionHeading
            eyebrow="Die Gala"
            title={
              <>
                Noch ein paar Runden <span className="text-gold">bis zur Verleihung.</span>
              </>
            }
            text="Zwei Tage vor Heiligabend, wie jedes Jahr, wird der Pokal übergeben. Wer zu spät kommt, hält die Laudatio."
          />
          <Reveal delay={0.15} className="mt-8 flex flex-col gap-3 text-sm text-foam-2">
            <p className="flex items-center gap-3">
              <CalendarDays size={18} className="text-gold" aria-hidden="true" />
              <time dateTime={site.ceremonyDate}>{site.ceremonyLabel}</time>
            </p>
            <p className="flex items-center gap-3">
              <MapPin size={18} className="text-gold" aria-hidden="true" />
              {site.ceremonyPlace}
            </p>
          </Reveal>
        </div>

        <Reveal delay={0.1}>
          <div
            className="grid grid-cols-4 gap-2 sm:gap-4"
            role="timer"
            aria-live="off"
            aria-label="Countdown bis zur Verleihung"
          >
            <Unit value={parts?.d ?? null} label="Tage" />
            <Unit value={parts?.h ?? null} label="Std" />
            <Unit value={parts?.m ?? null} label="Min" />
            <Unit value={parts?.s ?? null} label="Sek" />
          </div>
        </Reveal>
      </div>
    </section>
  );
}
