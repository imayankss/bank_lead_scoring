export function compactNumber(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

// MBD amounts are anonymised and noised, so values are shown without a currency.
export function currency(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(value);
}

export function interval(bounds?: [number, number], digits = 2) {
  if (!bounds || bounds.some((value) => value === undefined || value === null)) return "";
  return `95% CI ${bounds[0].toFixed(digits)}–${bounds[1].toFixed(digits)}`;
}

export function percent(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${value.toFixed(1)}%`;
}

export function decimal(value?: number, digits = 3) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}
