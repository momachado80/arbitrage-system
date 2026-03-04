import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        terminal: {
          bg: "#0B0F19",
          panel: "#141A2A",
          border: "#1E293B",
          text: "#E2E8F0",
          muted: "#94A3B8",
          green: "#4ADE80",
          yellow: "#FACC15",
          red: "#EF4444",
          cyan: "#22D3EE",
          blue: "#3B82F6",
        },
      },
      fontFamily: {
        mono: ["ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
