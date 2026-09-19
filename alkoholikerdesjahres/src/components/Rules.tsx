"use client";

import { motion } from "framer-motion";
import { rules } from "@/data/site";
import { SectionHeading, item, stagger } from "./Reveal";

export default function Rules() {
  return (
    <section id="regeln" className="section-pad relative scroll-mt-20">
      <div className="container-x">
        <SectionHeading
          eyebrow="So funktioniert's"
          title={
            <>
              Vier Regeln. <span className="text-gold">Keine Ausreden.</span>
            </>
          }
          text="Die Verleihung ist ernster als sie klingt und unernster als sie aussieht. Das hier ist die Kurzfassung der Satzung."
          align="center"
        />

        <motion.ol
          className="mt-14 grid gap-5 md:grid-cols-2 lg:grid-cols-4"
          variants={stagger}
          initial="hidden"
          whileInView="show"
          viewport={{ once: true, margin: "-10% 0px" }}
        >
          {rules.map((r, i) => (
            <motion.li
              key={r.step}
              variants={item}
              className="group relative overflow-hidden rounded-3xl glass p-6 transition-colors hover:border-gold/40"
            >
              <span
                className="pointer-events-none absolute -right-4 -top-6 font-display text-[7rem] font-black leading-none text-foam/[0.04] transition-transform duration-700 group-hover:-translate-y-2 group-hover:text-gold/10"
                aria-hidden="true"
              >
                {r.step}
              </span>
              <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-gold font-display text-sm font-extrabold text-ink">
                {i + 1}
              </span>
              <h3 className="mt-6 font-display text-xl font-bold">{r.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-foam-2">{r.text}</p>
            </motion.li>
          ))}
        </motion.ol>
      </div>
    </section>
  );
}
