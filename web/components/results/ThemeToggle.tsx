"use client";

import { useEffect, useState } from "react";

export function ThemeToggle() {
  const [theme, setTheme] = useState<string | null>(null);

  useEffect(() => setTheme(document.documentElement.dataset.theme ?? "dark"), []);

  const flip = () => {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try {
      localStorage.setItem("theme", next);
    } catch {}
    setTheme(next);
  };

  return (
    <button type="button" className="toggle" onClick={flip} aria-label="Switch colour theme">
      {theme === "dark" ? "Light" : "Dark"}
    </button>
  );
}
