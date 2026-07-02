import "katex/dist/katex.min.css";
import type { JSX } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import type { FilePreviewContentProps } from "@/lib/file-preview-plugins";
import { normalizeDisplayMath } from "@/lib/markdown-math";

export const MarkdownPreview = ({ content }: FilePreviewContentProps): JSX.Element => {
  return (
    <article className="prose prose-sm max-w-none px-4 py-3 [&_.katex-display]:overflow-x-auto">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[[rehypeKatex, { strict: "ignore" }]]}
      >
        {normalizeDisplayMath(content)}
      </ReactMarkdown>
    </article>
  );
};
