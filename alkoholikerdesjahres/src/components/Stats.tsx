"use client";

import { animate, motion, useInView, useReducedMotion } from "framer-motion";
import { useEffect, useRef, useState } from "react";
import { stats } from "@/data/site";
import { item, stagger } from "./Reveal";

function Counter({ value, suffix }: { value: number; suffix: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-20% 0px" });
  const reduced = useReducedMotion();
  const [n, setN] = useState(reduced ? value : 0);

  useEffect(() => {
    if (!inView || reduced) return;
    const controls = animate(0, value, {
      duration: 1.8,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (v) => setN(Math.round(v)),
    });
    return () => controls.stop();
  }, [inView, value, reduced]);

  return (
    <span ref={ref} className="tabular-nums">
      {n}
      {suffix}
    </span>
  );
}

export default function Stats() {
  return (
    <section className="relative border-y border-line bg-ink-2/40">
      <motion.dl
        className="container-x grid grid-cols-2 divide-line md:grid-cols-4 md:divide-x"
        variants={stagger}
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, margin: "-15% 0px" }}
      >
        {stats.map((s) => (
          <motion.div key={s.label} variants={item} className="px-2 py-10 md:px-8 md:py-14">
            <dd className="font-display text-4xl font-extrabold text-gold-gradient sm:text-5xl md:text-6xl">
              <Counter value={s.value} suffix={s.suffix} />
            </dd>
            <dt className="mt-2 text-sm text-foam-2">{s.label}</dt>
          </motion.div>
        ))}
      </motion.dl>
    </section>
  );
}
