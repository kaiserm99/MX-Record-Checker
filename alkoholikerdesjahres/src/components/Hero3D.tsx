"use client";

import dynamic from "next/dynamic";

const Scene = dynamic(() => import("./Scene"), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 grid place-items-center">
      <div className="relative h-24 w-24">
        <span className="absolute inset-0 rounded-full bg-gold/30 animate-pulse-ring" />
        <span className="absolute inset-6 rounded-full bg-gold/60 blur-md" />
      </div>
    </div>
  ),
});

export default function Hero3D() {
  return (
    <div className="absolute inset-0" aria-hidden="true">
      <Scene />
    </div>
  );
}
