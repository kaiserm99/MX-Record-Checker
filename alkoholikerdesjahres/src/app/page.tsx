import Countdown from "@/components/Countdown";
import FAQ from "@/components/FAQ";
import Footer from "@/components/Footer";
import HallOfFame from "@/components/HallOfFame";
import Hero from "@/components/Hero";
import Marquee from "@/components/Marquee";
import Navbar from "@/components/Navbar";
import Nominate from "@/components/Nominate";
import Nominees from "@/components/Nominees";
import Rules from "@/components/Rules";
import Stats from "@/components/Stats";

export default function Home() {
  return (
    <>
      <a
        href="#nominierte"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[100] focus:rounded-lg focus:bg-gold focus:px-4 focus:py-2 focus:text-ink"
      >
        Zum Inhalt springen
      </a>
      <Navbar />
      <main>
        <Hero />
        <Marquee />
        <Countdown />
        <Stats />
        <Nominees />
        <HallOfFame />
        <Rules />
        <Nominate />
        <FAQ />
      </main>
      <Footer />
    </>
  );
}
