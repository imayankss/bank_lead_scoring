import type { ReactNode } from "react";

export function SectionHeader({
  eyebrow,
  title,
  children
}: {
  eyebrow?: string;
  title: string;
  children?: ReactNode;
}) {
  return (
    <div className="mb-5">
      {eyebrow && <p className="mb-2 text-xs uppercase tracking-[0.24em] text-cyan">{eyebrow}</p>}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <h2 className="text-2xl font-semibold text-white md:text-3xl">{title}</h2>
        {children && <div className="max-w-2xl text-sm leading-6 text-slate-400">{children}</div>}
      </div>
    </div>
  );
}
