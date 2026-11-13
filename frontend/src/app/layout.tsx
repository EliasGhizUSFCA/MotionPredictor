import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MotionPredict AI",
  description: "Real-time pedestrian tracking and probabilistic trajectory forecasting with measurable accuracy.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
