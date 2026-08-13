export type DuplicateRegistrationPolicy = "throw" | "skip" | "replace";

export class ContributionRegistry<T extends { id: string }> {
  private items = new Map<string, T>();

  constructor(private readonly label: string) {}

  register(item: T, options: { onDuplicate?: DuplicateRegistrationPolicy } = {}): void {
    const { onDuplicate = "throw" } = options;

    if (this.items.has(item.id)) {
      if (onDuplicate === "skip") {
        return;
      }
      if (onDuplicate === "replace") {
        this.items.set(item.id, item);
        return;
      }
      throw new Error(`${this.label} "${item.id}" is already registered.`);
    }

    this.items.set(item.id, item);
  }

  unregister(id: string): boolean {
    return this.items.delete(id);
  }

  getAll(): T[] {
    return Array.from(this.items.values());
  }

  clear(): void {
    this.items.clear();
  }
}
