import type { Metadata } from "next";
import { Instrument_Sans, Instrument_Serif } from "next/font/google";

import {
  HEADLINE,
  PRODUCT_NAME,
  SUBHEAD,
} from "@bos/card-schema";

import "./globals.css";

const sans = Instrument_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

const serif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: PRODUCT_NAME,
  description: `${HEADLINE} ${SUBHEAD}`,
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${sans.variable} ${serif.variable}`}>
      <body>{children}</body>
    </html>
  );
}
