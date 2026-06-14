import type { JSX } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

/**
 * Token-styled markdown body for agent prose (answers, summaries, plans).
 *
 * The project does not ship the Tailwind typography plugin, so every
 * element is styled explicitly through descendant selectors against the
 * molcrafts theme tokens — same fidelity as the CLI's rich renderer:
 * fenced code in bordered muted blocks, GFM tables with hairline grids,
 * tight editor-density rhythm throughout.
 */

const MARKDOWN_CLASS = [
  "text-sm leading-relaxed text-foreground [overflow-wrap:anywhere]",
  // paragraphs + vertical rhythm
  "[&_p]:my-2 [&_>*:first-child]:mt-0 [&_>*:last-child]:mb-0",
  // headings — compact editor scale, weight carries hierarchy
  "[&_h1]:mt-4 [&_h1]:mb-1.5 [&_h1]:text-base [&_h1]:font-semibold",
  "[&_h2]:mt-3.5 [&_h2]:mb-1 [&_h2]:text-[0.9375rem] [&_h2]:font-semibold",
  "[&_h3]:mt-3 [&_h3]:mb-1 [&_h3]:text-sm [&_h3]:font-semibold",
  "[&_h4]:mt-2.5 [&_h4]:mb-0.5 [&_h4]:text-sm [&_h4]:font-medium",
  // lists
  "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-5 [&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-5",
  "[&_li]:my-0.5 [&_li_>_ul]:my-1 [&_li_>_ol]:my-1",
  // inline code
  "[&_:not(pre)>code]:rounded-sm [&_:not(pre)>code]:bg-muted [&_:not(pre)>code]:px-1",
  "[&_:not(pre)>code]:py-px [&_:not(pre)>code]:font-mono [&_:not(pre)>code]:text-[0.85em]",
  // fenced code blocks
  "[&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-md [&_pre]:border",
  "[&_pre]:border-border/60 [&_pre]:bg-muted/50 [&_pre]:px-3 [&_pre]:py-2.5",
  "[&_pre]:font-mono [&_pre]:text-xs [&_pre]:leading-relaxed",
  // links
  "[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2 hover:[&_a]:opacity-80",
  // quotes — conventional muted markdown affordance
  "[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border",
  "[&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground",
  // GFM tables
  "[&_table]:my-2 [&_table]:w-full [&_table]:border-collapse [&_table]:text-xs",
  "[&_th]:border [&_th]:border-border/60 [&_th]:bg-muted/40 [&_th]:px-2 [&_th]:py-1",
  "[&_th]:text-left [&_th]:font-medium",
  "[&_td]:border [&_td]:border-border/60 [&_td]:px-2 [&_td]:py-1 [&_td]:tabular-nums",
  // misc
  "[&_hr]:my-3 [&_hr]:border-border [&_strong]:font-semibold",
  "[&_img]:my-2 [&_img]:max-w-full [&_img]:rounded-md [&_img]:border [&_img]:border-border/60",
].join(" ");

export const MarkdownContent = ({
  text,
  className,
}: {
  text: string;
  className?: string;
}): JSX.Element => (
  <div className={cn(MARKDOWN_CLASS, className)}>
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ children, href }) => (
          <a href={href} target="_blank" rel="noreferrer">
            {children}
          </a>
        ),
      }}
    >
      {text}
    </ReactMarkdown>
  </div>
);
