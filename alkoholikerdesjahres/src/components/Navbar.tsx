"use client";

import { motion, useScroll, useSpring, AnimatePresence } from "framer-motion";
import { useEffect, useState } from "react";
import { Menu, X } from "lucide-react";
import { site } from "@/data/site";

const links = [
  { href: "#nominierte", label: "Nominierte" },
  { href: "#hall-of-fame", label: "Hall of Fame" },
  { href: "#regeln", label: "Regeln" },
  { href: "#faq", label: "FAQ" },
];

export default function Navbar() {
  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 24, mass: 0.4 });
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = open ? "hidden" : "";
    return () => {
      document.body.style.overflow = "";
    };
  }, [open]);

  return (
    <header className="fixed inset-x-0 top-0 z-50">
      <motion.div
        className="absolute inset-x-0 top-0 h-0.5 origin-left bg-gradient-to-r from-gold-3 via-gold to-gold-2"
        style={{ scaleX: progress }}
        aria-hidden="true"
      />
      <div className="container-x">
        <nav
          className={`mt-3 flex items-center justify-between rounded-2xl px-4 py-2.5 transition-all duration-500 sm:px-5 ${
            scrolled ? "glass shadow-card" : "border border-transparent"
          }`}
          aria-label="Hauptnavigation"
        >
          <a href="#top" className="group flex items-center gap-3">
            <span className="relative grid h-9 w-11 place-items-center rounded-xl bg-gradient-to-br from-gold-2 via-gold to-gold-3 font-display text-[11px] font-extrabold tracking-tight text-ink shadow-glow">
              {site.short}
              <span className="absolute -inset-1 -z-10 rounded-xl bg-gold/30 blur-md opacity-0 transition-opacity group-hover:opacity-100" />
            </span>
            <span className="hidden font-display text-sm font-semibold tracking-tight sm:block">
              {site.name} <span className="text-foam-3">{site.year}</span>
            </span>
          </a>

          <ul className="hidden items-center gap-1 md:flex">
            {links.map((l) => (
              <li key={l.href}>
                <a
                  href={l.href}
                  className="relative rounded-lg px-3 py-2 text-sm text-foam-2 transition-colors hover:text-foam after:absolute after:inset-x-3 after:-bottom-0.5 after:h-px after:origin-left after:scale-x-0 after:bg-gold after:transition-transform hover:after:scale-x-100"
                >
                  {l.label}
                </a>
              </li>
            ))}
          </ul>

          <div className="flex items-center gap-2">
            <a
              href="#nominieren"
              className="hidden rounded-xl bg-gold px-4 py-2 text-sm font-semibold text-ink transition-transform hover:-translate-y-0.5 hover:shadow-glow active:translate-y-0 md:inline-flex"
            >
              Jetzt nominieren
            </a>
            <button
              type="button"
              onClick={() => setOpen((v) => !v)}
              className="grid h-10 w-10 place-items-center rounded-xl text-foam md:hidden"
              aria-expanded={open}
              aria-controls="mobile-menu"
              aria-label={open ? "Menü schließen" : "Menü öffnen"}
            >
              {open ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </nav>
      </div>

      <AnimatePresence>
        {open && (
          <motion.div
            id="mobile-menu"
            className="fixed inset-0 top-0 z-40 flex flex-col bg-ink/95 px-6 pt-24 backdrop-blur-xl md:hidden"
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -12 }}
            transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
          >
            <ul className="flex flex-col gap-2">
              {[...links, { href: "#nominieren", label: "Jetzt nominieren" }].map((l, i) => (
                <motion.li
                  key={l.href}
                  initial={{ opacity: 0, x: -16 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.05 * i + 0.1 }}
                >
                  <a
                    href={l.href}
                    onClick={() => setOpen(false)}
                    className="block rounded-xl px-4 py-4 font-display text-2xl font-semibold text-foam hover:bg-foam/5"
                  >
                    {l.label}
                  </a>
                </motion.li>
              ))}
            </ul>
            <p className="mt-auto pb-10 text-sm text-foam-3">{site.tagline}</p>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  );
}
