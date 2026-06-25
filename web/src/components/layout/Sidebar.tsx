"use client";

import {
  Activity,
  BarChart3,
  BrainCircuit,
  Gauge,
  GitBranch,
  Layers3,
  LineChart,
  Microscope,
  Radar,
  Sparkles
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";

const navItems = [
  { href: "/", label: "Overview", icon: Gauge },
  { href: "/forecast", label: "Forecast", icon: LineChart },
  { href: "/models", label: "Models", icon: BrainCircuit },
  { href: "/risk", label: "Risk", icon: Radar },
  { href: "/diagnostics", label: "Diagnostics", icon: Microscope },
  { href: "/seasonality", label: "Seasonality", icon: Activity },
  { href: "/scenario", label: "Scenario", icon: GitBranch },
  { href: "/methodology", label: "Methodology", icon: Layers3 }
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="fixed inset-y-0 left-0 z-30 hidden w-72 border-r border-line bg-ink/90 px-5 py-6 backdrop-blur-xl lg:block">
      <Link href="/" className="mb-8 flex items-center gap-3">
        <div className="flex h-11 w-11 items-center justify-center rounded-lg border border-cyan/30 bg-cyan/10">
          <Sparkles className="h-5 w-5 text-cyan" />
        </div>
        <div>
          <div className="text-sm uppercase tracking-[0.22em] text-slate-400">AI Command</div>
          <div className="text-lg font-semibold text-white">Lead Intelligence</div>
        </div>
      </Link>
      <nav className="space-y-2">
        {navItems.map((item) => {
          const active = pathname === item.href || (item.href !== "/" && pathname.startsWith(item.href));
          const Icon = item.icon;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-3 rounded-lg px-3 py-3 text-sm text-slate-300 transition",
                active ? "border border-cyan/30 bg-cyan/10 text-white shadow-glow" : "hover:bg-white/6 hover:text-white"
              )}
            >
              <Icon className="h-4 w-4" />
              {item.label}
            </Link>
          );
        })}
      </nav>
      <div className="absolute bottom-6 left-5 right-5 rounded-lg border border-line bg-white/[0.03] p-4">
        <div className="mb-2 flex items-center gap-2 text-sm font-medium text-white">
          <BarChart3 className="h-4 w-4 text-mint" />
          Static ML Export
        </div>
        <p className="text-xs leading-5 text-slate-400">
          Python owns the scoring logic. This dashboard reads generated JSON contracts only.
        </p>
      </div>
    </aside>
  );
}
