import type { Metadata } from "next";
import { IBM_Plex_Mono, Instrument_Sans, Schibsted_Grotesk } from "next/font/google";
import "./globals.css";

// Everything read: nav, list rows, email bodies. A grotesk with a slightly
// narrow, technical cut that stays crisp at 13-14px list sizes.
const instrumentSans = Instrument_Sans({
  variable: "--font-sans",
  subsets: ["latin"],
});

// Headings only (mailbox titles, email subjects, section heads). Used with
// restraint so it reads as the app's voice rather than decoration.
const schibstedGrotesk = Schibsted_Grotesk({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "600", "700"],
});

// Numbers only: times, counts, and model confidences.
const plexMono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "Inbox",
  description: "An inbox that sorts your mail on your own device.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${instrumentSans.variable} ${schibstedGrotesk.variable} ${plexMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
