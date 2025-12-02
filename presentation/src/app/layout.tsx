import type { Metadata } from "next";
import { Inter, Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
});

const spaceGrotesk = Space_Grotesk({
  variable: "--font-space-grotesk",
  subsets: ["latin"],
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MIS587 Final Project | ML for Real Estate Valuation",
  description: "Massachusetts Industrial Properties Price Prediction and Market Analysis - A Machine Learning Approach by Team 2",
  keywords: ["Machine Learning", "Real Estate", "Property Valuation", "Neural Network", "SHAP Analysis"],
  authors: [
    { name: "Alex Siracusa" },
    { name: "Martin Thulani Milanzi" },
    { name: "Shrey Sharma" },
    { name: "Faisal Yaseen" },
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body
        className={`${inter.variable} ${spaceGrotesk.variable} ${jetbrainsMono.variable} font-sans antialiased`}
      >
        <div className="fixed inset-0 mesh-bg pointer-events-none" />
        <div className="fixed inset-0 grid-pattern pointer-events-none opacity-50" />
        {children}
      </body>
    </html>
  );
}
