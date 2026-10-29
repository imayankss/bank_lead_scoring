// Shared Recharts styling, driven by the CSS tokens so both themes work.
export const axis = { stroke: "var(--axis)", tick: { fill: "var(--text-muted)", fontSize: 11, fontFamily: "var(--mono)" } };
export const grid = { stroke: "var(--grid)", strokeDasharray: "0" };
export const tooltip = {
  contentStyle: { background: "var(--surface)", border: "1px solid var(--rule)", borderRadius: 0, fontFamily: "var(--mono)", fontSize: 12 },
  labelStyle: { color: "var(--text-muted)" },
  itemStyle: { color: "var(--text-primary)" }
};
