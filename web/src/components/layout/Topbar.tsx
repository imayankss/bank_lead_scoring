import { CircleDot, Database, ShieldCheck } from "lucide-react";
import Link from "next/link";

const mobileLinks = [
  ["Overview", "/"],
  ["Forecast", "/forecast"],
  ["Models", "/models"],
  ["Risk", "/risk"],
  ["Diagnostics", "/diagnostics"],
  ["Scenario", "/scenario"]
];

export function Topbar() {
  return (
    <header className="sticky top-0 z-20 border-b border-line bg-ink/72 px-4 py-3 backdrop-blur-xl lg:ml-72 lg:px-8">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.22em] text-slate-500">Lead Scoring Intelligence Platform</p>
          <h1 className="text-xl font-semibold text-white">Banking Customer Analytics</h1>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-white/[0.04] px-3 py-2">
            <CircleDot className="h-3.5 w-3.5 text-mint" />
            Pipeline verified
          </span>
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-white/[0.04] px-3 py-2">
            <Database className="h-3.5 w-3.5 text-cyan" />
            Static JSON
          </span>
          <span className="inline-flex items-center gap-2 rounded-full border border-line bg-white/[0.04] px-3 py-2">
            <ShieldCheck className="h-3.5 w-3.5 text-amber" />
            Leakage guarded
          </span>
        </div>
      </div>
      <nav className="mt-3 flex gap-2 overflow-x-auto pb-1 lg:hidden">
        {mobileLinks.map(([label, href]) => (
          <Link
            key={href}
            href={href}
            className="shrink-0 rounded-full border border-line bg-white/[0.04] px-3 py-2 text-xs text-slate-300"
          >
            {label}
          </Link>
        ))}
      </nav>
    </header>
  );
}
