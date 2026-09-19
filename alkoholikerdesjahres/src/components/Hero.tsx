"use client";

import { motion, useReducedMotion, useScroll, useTransform } from "framer-motion";
import { ArrowDown, Sparkles as SparkIcon } from "lucide-react";
import { useRef } from "react";
import Hero3D from "./Hero3D";
import { site } from "@/data/site";

const ease = [0.22, 1, 0.36, 1] as const;

function Word({ children, i }: { children: string; i: number }) {
  return (
    <span className="-mx-[0.06em] inline-block overflow-hidden px-[0.06em] pb-[0.08em] align-bottom">
      <motion.span
        className="inline-block"
        initial={{ y: "110%", rotate: 4 }}
        animate={{ y: 0, rotate: 0 }}
        transition={{ duration: 0.9, ease, delay: 0.25 + i * 0.09 }}
      >
        {children}
      </motion.span>
    </span>
  );
}

export default function Hero() {
  const ref = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start start", "end start"] });
  const sceneY = useTransform(scrollYProgress, [0, 1], ["0%", reduced ? "0%" : "22%"]);
  const sceneOpacity = useTransform(scrollYProgress, [0, 0.75], [1, 0]);
  const copyY = useTransform(scrollYProgress, [0, 1], ["0%", reduced ? "0%" : "-12%"]);

  return (
    <section
      id="top"
      ref={ref}
      className="relative isolate flex min-h-[100svh] items-end overflow-hidden pb-12 pt-[46svh] sm:items-center sm:pb-20 sm:pt-28"
    >
      {/* background glows */}
      <div className="pointer-events-none absolute inset-0 -z-20" aria-hidden="true">
        <div className="absolute left-1/2 top-[-20%] h-[70vh] w-[70vh] -translate-x-1/2 rounded-full bg-gold/15 blur-[120px]" />
        <div className="absolute right-[-10%] bottom-[-10%] h-[50vh] w-[50vh] rounded-full bg-cherry/15 blur-[120px]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,transparent_40%,var(--color-ink)_100%)]" />
      </div>

      <motion.div
        style={{ y: sceneY, opacity: sceneOpacity }}
        className="absolute inset-x-0 top-0 -z-10 h-[64svh] sm:inset-0 sm:h-auto lg:left-[30%]"
      >
        <Hero3D />
      </motion.div>
      {/* keeps the copy readable over the scene on small screens */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-x-0 bottom-0 -z-10 h-[70%] bg-gradient-to-t from-ink via-ink/90 to-transparent sm:hidden"
      />

      <motion.div style={{ y: copyY }} className="container-x relative">
        <div className="max-w-3xl">
          <motion.p
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, ease, delay: 0.1 }}
            className="mb-6 inline-flex items-center gap-2 whitespace-nowrap rounded-full glass px-3.5 py-1.5 text-[10px] font-medium uppercase tracking-[0.14em] text-gold sm:text-xs sm:tracking-[0.2em]"
          >
            <SparkIcon size={14} aria-hidden="true" />
            <span className="sm:hidden">Verleihung {site.year} · Satire</span>
            <span className="hidden sm:inline">Die Verleihung {site.year} · Satire mit Herz</span>
          </motion.p>

          <h1 className="font-display text-[clamp(2.6rem,9vw,7rem)] font-extrabold leading-[0.95] tracking-tight">
            <Word i={0}>Alkoholiker</Word>{" "}
            <br className="hidden sm:block" />
            <span className="text-gold-gradient animate-shimmer">
              <Word i={1}>des</Word> <Word i={2}>Jahres</Word>
            </span>
          </h1>

          <motion.p
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease, delay: 0.7 }}
            className="mt-7 max-w-xl text-balance text-lg leading-relaxed text-foam-2 md:text-xl"
          >
            {site.tagline} Ein Wanderpokal unter Freund:innen für Ausdauer, Stil und die besten
            Geschichten vom Tresen.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease, delay: 0.85 }}
            className="mt-9 flex flex-wrap items-center gap-3"
          >
            <a
              href="#nominieren"
              className="group relative inline-flex items-center gap-2 overflow-hidden rounded-2xl bg-gold px-6 py-3.5 font-semibold text-ink shadow-glow transition-transform hover:-translate-y-0.5 active:translate-y-0"
            >
              <span className="absolute inset-0 -translate-x-full bg-gradient-to-r from-transparent via-white/50 to-transparent transition-transform duration-700 group-hover:translate-x-full" />
              <span className="relative">Jemanden nominieren</span>
            </a>
            <a
              href="#hall-of-fame"
              className="inline-flex items-center gap-2 rounded-2xl glass px-6 py-3.5 font-semibold text-foam transition-colors hover:border-gold/40 hover:text-gold"
            >
              Zur Hall of Fame
            </a>
          </motion.div>

          <motion.dl
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 1, delay: 1.1 }}
            className="mt-12 grid max-w-lg grid-cols-3 gap-4 border-t border-line pt-6 text-sm"
          >
            {[
              ["Wann", "Silvester"],
              ["Wo", "Hinterzimmer"],
              ["Dresscode", "Standfest"],
            ].map(([k, v]) => (
              <div key={k}>
                <dt className="text-xs uppercase tracking-widest text-foam-3">{k}</dt>
                <dd className="mt-1 font-display font-semibold text-foam">{v}</dd>
              </div>
            ))}
          </motion.dl>
        </div>
      </motion.div>

      <motion.a
        href="#countdown"
        aria-label="Nach unten scrollen"
        className="absolute bottom-6 left-1/2 hidden -translate-x-1/2 flex-col items-center gap-2 text-xs uppercase tracking-widest text-foam-3 sm:flex"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.6 }}
      >
        <span>Scroll</span>
        <motion.span
          animate={reduced ? undefined : { y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 1.6, ease: "easeInOut" }}
        >
          <ArrowDown size={16} />
        </motion.span>
      </motion.a>
    </section>
  );
}
