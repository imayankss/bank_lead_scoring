import { decimal } from "@/lib/formatters";
import type { ModelRecord } from "@/lib/types";
import { ModelBadge } from "./ModelBadge";

export function ModelLeaderboard({ models }: { models: ModelRecord[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-line">
      <table className="w-full text-left text-sm">
        <thead className="bg-white/[0.04] text-xs uppercase tracking-[0.16em] text-slate-500">
          <tr>
            <th className="px-4 py-3">Rank</th>
            <th className="px-4 py-3">Model</th>
            <th className="px-4 py-3">Business Score</th>
            <th className="px-4 py-3">AUC</th>
            <th className="px-4 py-3">RMSE</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-line">
          {models.map((model) => (
            <tr key={model.model_id} className="bg-white/[0.02]">
              <td className="px-4 py-4 text-slate-300">#{model.rank ?? "-"}</td>
              <td className="px-4 py-4">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="font-medium text-white">{model.model_name ?? model.display_name}</span>
                  {model.is_best && <ModelBadge />}
                </div>
              </td>
              <td className="px-4 py-4 text-mint">{decimal(model.business_score, 3)}</td>
              <td className="px-4 py-4 text-slate-300">{decimal(model.auc, 3)}</td>
              <td className="px-4 py-4 text-slate-300">{decimal(model.rmse, 0)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
