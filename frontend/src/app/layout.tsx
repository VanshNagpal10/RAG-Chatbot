import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CHROMA",
  description:
    "Compare social media videos with AI-powered transcript analysis. " +
    "Ingest YouTube & Instagram videos, compute engagement metrics, and " +
    "chat with an AI that understands your content.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body className="bg-zinc-950 text-zinc-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}
