/**
 * Count + noun label with naive s-pluralization — "1 run" / "3 runs".
 * Enough for the sidebar count chips; pass an explicit plural for nouns
 * that don't pluralize with a trailing "s".
 */
export const countLabel = (count: number, noun: string, plural: string = `${noun}s`): string =>
  `${count} ${count === 1 ? noun : plural}`;
