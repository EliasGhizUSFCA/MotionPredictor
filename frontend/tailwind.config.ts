import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0f14",
        panel: "#10161e",
        panel2: "#151d28",
        line: "#233041",
        ink: "#e2e8f0",
        dim: "#8b9bb0",
        accent: "#22d3ee",
        good: "#22c55e",
        warn: "#eab308",
        bad: "#f43f5e",
        obs: "#3b82f6",
      },
      fontFamily: { mono: ["ui-monospace", "SFMono-Regular", "Menlo", "monospace"] },
    },
  },
  plugins: [],
} satisfies Config;
