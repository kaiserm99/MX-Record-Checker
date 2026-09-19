"use client";

import { motion, useMotionValue, useReducedMotion, useSpring, useTransform } from "framer-motion";
import { Heart, Quote } from "lucide-react";
import { useMemo, useSyncExternalStore, type MouseEvent } from "react";
import { nominees, type Nominee } from "@/data/site";
import { SectionHeading, item, stagger } from "./Reveal";

const STORAGE_KEY = "adj-votes-2026";

const VOTE_EVENT = "adj-votes-change";

let memoryRaw = "{}"; // fallback when storage is blocked (private mode, disabled site data)

function readRaw() {
  try {
    return localStorage.getItem(STORAGE_KEY) ?? memoryRaw;
  } catch {
    return memoryRaw;
  }
}
function subscribe(cb: () => void) {
  window.addEventListener(VOTE_EVENT, cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener(VOTE_EVENT, cb);
    window.removeEventListener("storage", cb);
  };
}
const getServerRaw = () => "{}";

/** Votes live only in this browser. The store is read through useSyncExternalStore so SSR and hydration match. */
function useVotes() {
  const raw = useSyncExternalStore(subscribe, readRaw, getServerRaw);
  const votes = useMemo<Record<string, boolean>>(() => {
    try {
      return JSON.parse(raw);
    } catch {
      return {};
    }
  }, [raw]);
  const toggle = (id: string) => {
    const next = { ...votes, [id]: !votes[id] };
    memoryRaw = JSON.stringify(next);
    try {
      localStorage.setItem(STORAGE_KEY, memoryRaw);
    } catch {}
    window.dispatchEvent(new Event(VOTE_EVENT));
  };
  return { votes, toggle };
}

function Card({ n, voted, onVote, index }: { n: Nominee; voted: boolean; onVote: () => void; index: number }) {
  const reduced = useReducedMotion();
  const mx = useMotionValue(0.5);
  const my = useMotionValue(0.5);
  const rx = useSpring(useTransform(my, [0, 1], [9, -9]), { stiffness: 200, damping: 20 });
  const ry = useSpring(useTransform(mx, [0, 1], [-11, 11]), { stiffness: 200, damping: 20 });
  const glowX = useTransform(mx, [0, 1], ["0%", "100%"]);
  const glowY = useTransform(my, [0, 1], ["0%", "100%"]);

  const onMove = (e: MouseEvent<HTMLElement>) => {
    if (reduced) return;
    const r = e.currentTarget.getBoundingClientRect();
    mx.set((e.clientX - r.left) / r.width);
    my.set((e.clientY - r.top) / r.height);
  };
  const onLeave = () => {
    mx.set(0.5);
    my.set(0.5);
  };

  return (
    <motion.li variants={item} className="[perspective:1200px]">
      <motion.article
        onMouseMove={onMove}
        onMouseLeave={onLeave}
        style={{ rotateX: rx, rotateY: ry, transformStyle: "preserve-3d" }}
        className="group relative flex h-full flex-col rounded-3xl glass p-6 shadow-card transition-colors hover:border-gold/40"
      >
        <motion.div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0 rounded-3xl opacity-0 transition-opacity duration-500 group-hover:opacity-100"
          style={{
            background: `radial-gradient(420px circle at var(--gx) var(--gy), hsl(${n.hue} 90% 60% / 0.18), transparent 45%)`,
            // @ts-expect-error CSS custom properties are fine on style
            "--gx": glowX,
            "--gy": glowY,
          }}
        />

        <div className="flex items-start justify-between gap-4" style={{ transform: "translateZ(30px)" }}>
          <div
            className="grid h-16 w-16 shrink-0 place-items-center rounded-2xl font-display text-xl font-extrabold text-ink shadow-lg"
            style={{
              background: `linear-gradient(135deg, hsl(${n.hue} 90% 72%), hsl(${n.hue} 80% 48%))`,
            }}
            aria-hidden="true"
          >
            {n.initials}
          </div>
          <span className="rounded-full border border-line px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-gold">
            {n.category}
          </span>
        </div>

        <h3 className="mt-6 font-display text-2xl font-bold" style={{ transform: "translateZ(24px)" }}>
          {n.nick}
        </h3>
        <p className="mt-2 text-sm leading-relaxed text-foam-2">{n.title}</p>

        <blockquote className="mt-5 flex gap-2 rounded-2xl bg-ink/50 p-4 text-sm italic text-foam">
          <Quote size={16} className="shrink-0 text-gold" aria-hidden="true" />
          <p>{n.quote}</p>
        </blockquote>

        <dl className="mt-5 text-xs">
          <dt className="uppercase tracking-[0.2em] text-foam-3">Signature Drink</dt>
          <dd className="mt-1 text-sm text-foam-2">{n.drink}</dd>
        </dl>

        <div className="mt-auto flex items-center justify-between pt-6">
          <span className="text-xs text-foam-3">#{String(index + 1).padStart(2, "0")}</span>
          <button
            type="button"
            onClick={onVote}
            aria-pressed={voted}
            className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-semibold transition-all ${
              voted
                ? "bg-cherry text-foam shadow-[0_0_30px_-8px_var(--color-cherry)]"
                : "bg-foam/10 text-foam hover:bg-foam/15"
            }`}
          >
            <motion.span
              animate={voted && !reduced ? { scale: [1, 1.5, 1] } : { scale: 1 }}
              transition={{ duration: 0.4 }}
              className="inline-flex"
            >
              <Heart size={16} fill={voted ? "currentColor" : "none"} aria-hidden="true" />
            </motion.span>
            {voted ? "Gedrückt" : "Daumen drücken"}
          </button>
        </div>
      </motion.article>
    </motion.li>
  );
}

export default function Nominees() {
  const { votes, toggle } = useVotes();
  const count = Object.values(votes).filter(Boolean).length;

  return (
    <section id="nominierte" className="section-pad relative scroll-mt-20">
      <div className="container-x">
        <div className="flex flex-col gap-6 md:flex-row md:items-end md:justify-between">
          <SectionHeading
            eyebrow={`Nominierte ${new Date().getFullYear()}`}
            title={
              <>
                Sechs Legenden. <span className="text-gold">Ein Pokal.</span>
              </>
            }
            text="Die Jury hat vorsortiert. Du kannst Daumen drücken, so viele du willst. Die Entscheidung fällt trotzdem im Hinterzimmer."
          />
          <p className="text-sm text-foam-3" aria-live="polite">
            {count === 0 ? "Noch niemandem die Daumen gedrückt." : `${count} × Daumen gedrückt.`}
          </p>
        </div>

        <motion.ul
          className="mt-12 grid gap-5 sm:grid-cols-2 lg:grid-cols-3"
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-10% 0px" }}
        >
          {nominees.map((n, i) => (
            <Card key={n.id} n={n} index={i} voted={!!votes[n.id]} onVote={() => toggle(n.id)} />
          ))}
        </motion.ul>
      </div>
    </section>
  );
}
