import type { LucideIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export function MetricCard({
  label,
  value,
  detail,
  icon: Icon,
  tone = "cyan"
}: {
  label: string;
  value: string;
  detail?: string;
  icon: LucideIcon;
  tone?: "cyan" | "mint" | "amber" | "rose";
}) {
  const tones = {
    cyan: "text-cyan bg-cyan/10 border-cyan/25",
    mint: "text-mint bg-mint/10 border-mint/25",
    amber: "text-amber bg-amber/10 border-amber/25",
    rose: "text-rose bg-rose/10 border-rose/25"
  };

  return (
    <div className="glass rounded-lg p-5">
      <div className="mb-4 flex items-center justify-between gap-3">
        <p className="text-sm text-slate-400">{label}</p>
        <div className={cn("rounded-lg border p-2", tones[tone])}>
          <Icon className="h-4 w-4" />
        </div>
      </div>
      <div className="text-2xl font-semibold text-white">{value}</div>
      {detail && <p className="mt-2 text-sm text-slate-500">{detail}</p>}
    </div>
  );
}
