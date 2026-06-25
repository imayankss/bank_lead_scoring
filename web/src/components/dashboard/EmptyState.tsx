import { FileQuestion } from "lucide-react";

export function EmptyState({ title = "No data available", detail = "Run the backend export to refresh this panel." }) {
  return (
    <div className="flex min-h-56 flex-col items-center justify-center rounded-lg border border-dashed border-line bg-white/[0.02] p-8 text-center">
      <FileQuestion className="mb-3 h-8 w-8 text-slate-500" />
      <p className="font-medium text-slate-200">{title}</p>
      <p className="mt-2 text-sm text-slate-500">{detail}</p>
    </div>
  );
}
