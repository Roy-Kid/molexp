import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { FilePreviewContentProps } from "@/lib/file-preview-plugins";

export const MarkdownPreview = ({ content }: FilePreviewContentProps): JSX.Element => {
  return (
    <article className="prose prose-sm max-w-none px-4 py-3">
      <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
    </article>
  );
};
