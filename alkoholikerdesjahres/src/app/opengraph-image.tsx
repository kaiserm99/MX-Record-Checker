import { ImageResponse } from "next/og";
import { site } from "@/data/site";

export const dynamic = "force-static";
export const alt = `${site.name} ${site.year}`;
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default function OpenGraphImage() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: 72,
          background: "radial-gradient(circle at 20% 0%, #3a2a10 0%, #09070b 55%)",
          color: "#f8f1e4",
          fontFamily: "sans-serif",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 18, fontSize: 28, color: "#f2b233", letterSpacing: 6 }}>
          <div
            style={{
              width: 56,
              height: 56,
              borderRadius: 14,
              background: "linear-gradient(135deg,#ffd97a,#b8811a)",
              color: "#09070b",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontWeight: 800,
              fontSize: 22,
            }}
          >
            {site.short}
          </div>
          DIE VERLEIHUNG {site.year}
        </div>
        <div style={{ display: "flex", flexDirection: "column" }}>
          <div style={{ fontSize: 108, fontWeight: 900, lineHeight: 1 }}>Alkoholiker</div>
          <div
            style={{
              fontSize: 108,
              fontWeight: 900,
              lineHeight: 1,
              background: "linear-gradient(90deg,#ffd97a,#f2b233,#b8811a)",
              backgroundClip: "text",
              color: "transparent",
            }}
          >
            des Jahres
          </div>
          <div style={{ marginTop: 28, fontSize: 32, color: "#cdbfae" }}>{site.tagline}</div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", fontSize: 24, color: "#8f8375" }}>
          <span>{site.domain}</span>
          <span>Satire · Ab 18 · Trink verantwortungsvoll</span>
        </div>
      </div>
    ),
    size,
  );
}
