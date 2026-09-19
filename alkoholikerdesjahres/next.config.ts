import type { NextConfig } from "next";

// STATIC_EXPORT=1 npm run build → writes a fully static site to ./out (for hosts without a Node runtime)
const nextConfig: NextConfig = {
  output: process.env.STATIC_EXPORT ? "export" : undefined,
  images: { unoptimized: true },
};

export default nextConfig;
