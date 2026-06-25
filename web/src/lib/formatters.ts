export function compactNumber(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function currency(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `INR ${new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 2 }).format(value)}`;
}

export function percent(value?: number) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return `${value.toFixed(1)}%`;
}

export function decimal(value?: number, digits = 3) {
  if (value === undefined || Number.isNaN(value)) return "N/A";
  return value.toFixed(digits);
}
