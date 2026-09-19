"use client";

import { motion, useScroll, useSpring } from "framer-motion";
import { Trophy } from "lucide-react";
import { useRef } from "react";
import { winners } from "@/data/site";
import { Reveal, SectionHeading } from "./Reveal";

export default function HallOfFame() {
  const ref = useRef<HTMLOListElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ["start 75%", "end 60%"] });
  const line = useSpring(scrollYProgress, { stiffness: 80, damping: 22 });

  return (
    <section id="hall-of-fame" className="section-pad relative scroll-mt-20 bg-ink-2/40">
      <div
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(60%_50%_at_80%_20%,rgba(242,178,51,0.10),transparent)]"
      />
      <div className="container-x">
        <SectionHeading
          eyebrow="Hall of Fame"
          title={
            <>
              Wer den Pokal <span className="text-gold">schon getragen hat.</span>
            </>
          }
          text="Jede:r hier hat ihn ein Jahr lang auf dem Regal stehen gehabt. Manche auch woanders. Wir fragen nicht."
        />

        <ol ref={ref} className="relative mt-14 space-y-8 pl-8 sm:pl-12">
          <div className="absolute left-[9px] top-0 h-full w-px bg-line sm:left-[13px]" aria-hidden="true" />
          <motion.div
            className="absolute left-[9px] top-0 h-full w-px origin-top bg-gradient-to-b from-gold-2 via-gold to-transparent sm:left-[13px]"
            style={{ scaleY: line }}
            aria-hidden="true"
          />
          {winners.map((w, i) => (
            <li key={w.year} className="relative">
              <span
                className={`absolute -left-8 top-2 grid h-5 w-5 place-items-center rounded-full border-2 sm:-left-12 sm:h-7 sm:w-7 ${
                  w.highlight ? "border-gold bg-gold text-ink shadow-glow" : "border-line bg-ink text-gold"
                }`}
                aria-hidden="true"
              >
                <Trophy size={12} />
              </span>
              <Reveal delay={i * 0.04}>
                <article
                  className={`grid gap-3 rounded-3xl p-6 transition-colors sm:grid-cols-[8.5rem_1fr] sm:gap-8 ${
                    w.highlight ? "glass border-gold/30" : "hover:bg-foam/[0.03]"
                  }`}
                >
                  <p className="font-display text-3xl font-extrabold tabular-nums text-gold-gradient sm:text-4xl">{w.year}</p>
                  <div>
                    <h3 className="font-display text-xl font-bold">
                      {w.nick}
                      {w.highlight ? (
                        <span className="ml-3 rounded-full bg-gold px-2 py-0.5 align-middle text-[10px] font-bold uppercase tracking-widest text-ink">
                          Titelverteidiger
                        </span>
                      ) : null}
                    </h3>
                    <p className="mt-1.5 text-sm leading-relaxed text-foam-2">{w.reason}</p>
                  </div>
                </article>
              </Reveal>
            </li>
          ))}
        </ol>
      </div>
    </section>
  );
}
