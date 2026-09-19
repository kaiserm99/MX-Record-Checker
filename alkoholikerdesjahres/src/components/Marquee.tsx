import { marquee } from "@/data/site";

export default function Marquee() {
  const items = [...marquee, ...marquee];
  return (
    <div className="relative overflow-hidden border-y border-line bg-ink-2/60 py-4 mask-fade-x" aria-hidden="true">
      <div className="flex w-max animate-marquee gap-10 whitespace-nowrap">
        {items.map((w, i) => (
          <span
            key={`${w}-${i}`}
            className="flex items-center gap-10 font-display text-lg font-semibold uppercase tracking-[0.3em] text-foam-2"
          >
            {w}
            <span className="h-1.5 w-1.5 rounded-full bg-gold" />
          </span>
        ))}
      </div>
    </div>
  );
}
