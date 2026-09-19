"use client";

import { AnimatePresence, motion } from "framer-motion";
import { Plus } from "lucide-react";
import { useState } from "react";
import { faq } from "@/data/site";
import { Reveal, SectionHeading } from "./Reveal";

export default function FAQ() {
  const [open, setOpen] = useState<number | null>(0);

  return (
    <section id="faq" className="section-pad relative scroll-mt-20 bg-ink-2/40">
      <div className="container-x grid gap-12 lg:grid-cols-[1fr_1.4fr]">
        <SectionHeading
          eyebrow="FAQ"
          title={
            <>
              Fragen, die <span className="text-gold">jedes Jahr</span> kommen.
            </>
          }
          text="Und die Antworten, die jedes Jahr niemand liest."
        />

        <Reveal delay={0.1}>
          <ul className="divide-y divide-line rounded-3xl border border-line">
            {faq.map((f, i) => {
              const isOpen = open === i;
              return (
                <li key={f.q}>
                  <h3>
                    <button
                      type="button"
                      onClick={() => setOpen(isOpen ? null : i)}
                      aria-expanded={isOpen}
                      aria-controls={`faq-panel-${i}`}
                      id={`faq-btn-${i}`}
                      className="flex w-full items-center justify-between gap-6 px-6 py-5 text-left font-display text-lg font-semibold transition-colors hover:text-gold"
                    >
                      {f.q}
                      <motion.span
                        animate={{ rotate: isOpen ? 45 : 0 }}
                        transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
                        className={`grid h-8 w-8 shrink-0 place-items-center rounded-full border ${
                          isOpen ? "border-gold bg-gold text-ink" : "border-line text-foam-2"
                        }`}
                        aria-hidden="true"
                      >
                        <Plus size={16} />
                      </motion.span>
                    </button>
                  </h3>
                  <AnimatePresence initial={false}>
                    {isOpen && (
                      <motion.div
                        id={`faq-panel-${i}`}
                        role="region"
                        aria-labelledby={`faq-btn-${i}`}
                        initial={{ height: 0, opacity: 0 }}
                        animate={{ height: "auto", opacity: 1 }}
                        exit={{ height: 0, opacity: 0 }}
                        transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                        className="overflow-hidden"
                      >
                        <p className="px-6 pb-6 text-sm leading-relaxed text-foam-2 md:text-base">{f.a}</p>
                      </motion.div>
                    )}
                  </AnimatePresence>
                </li>
              );
            })}
          </ul>
        </Reveal>
      </div>
    </section>
  );
}
