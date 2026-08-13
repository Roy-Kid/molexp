export type AgentMode = "chat" | "plan";

export const nextAgentMode = (mode: AgentMode): AgentMode => (mode === "chat" ? "plan" : "chat");
