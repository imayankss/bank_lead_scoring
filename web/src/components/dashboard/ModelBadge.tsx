import { Trophy } from "lucide-react";

export function ModelBadge({ label = "Best model" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-2 rounded-full border border-mint/30 bg-mint/10 px-3 py-1 text-xs font-medium text-mint">
      <Trophy className="h-3.5 w-3.5" />
      {label}
    </span>
  );
}
