import type { ReactNode } from "react";

export function ChartCard({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="glass rounded-lg p-5">
      <h3 className="mb-4 text-base font-semibold text-white">{title}</h3>
      {children}
    </section>
  );
}
