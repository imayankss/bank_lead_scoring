import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#07090d",
        panel: "#11151d",
        line: "rgba(255,255,255,0.11)",
        cyan: "#22d3ee",
        mint: "#34d399",
        amber: "#f59e0b",
        rose: "#fb7185"
      },
      boxShadow: {
        glow: "0 0 40px rgba(34, 211, 238, 0.14)"
      }
    }
  },
  plugins: []
};

export default config;
