import { AlertTriangle, CheckCircle2 } from "lucide-react";

export function RiskBadge({ category }: { category: string }) {
  const high = category.toLowerCase() === "high";
  return (
    <span className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-medium ${high ? "border-rose/30 bg-rose/10 text-rose" : "border-mint/30 bg-mint/10 text-mint"}`}>
      {high ? <AlertTriangle className="h-3.5 w-3.5" /> : <CheckCircle2 className="h-3.5 w-3.5" />}
      {category}
    </span>
  );
}
