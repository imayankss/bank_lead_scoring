import { ArrowRight } from "lucide-react";

export function PipelineTimeline({ steps }: { steps: string[] }) {
  return (
    <div className="grid gap-3 md:grid-cols-3 xl:grid-cols-4">
      {steps.map((step, index) => (
        <div key={step} className="glass flex items-center gap-3 rounded-lg p-4">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-cyan/10 text-sm font-semibold text-cyan">
            {index + 1}
          </div>
          <div className="min-w-0 flex-1 text-sm font-medium text-slate-200">{step}</div>
          {index < steps.length - 1 && <ArrowRight className="hidden h-4 w-4 text-slate-600 xl:block" />}
        </div>
      ))}
    </div>
  );
}
