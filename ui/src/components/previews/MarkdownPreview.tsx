/**
 * Markdown Preview Component
 * 
 * Renders markdown content with GitHub-Flavored Markdown support.
 * Uses react-markdown with remark-gfm for proper rendering of:
 * - Headings, paragraphs, lists
 * - Code blocks with syntax highlighting
 * - Tables, strikethrough, autolinks
 * - Task lists
 */

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { ScrollArea } from '@/components/ui/scroll-area';
import type { FilePreviewContentProps } from '@/lib/file-preview-plugins';

/**
 * MarkdownPreview renders markdown content with proper styling.
 * Uses Tailwind typography classes for a consistent reading experience.
 */
export const MarkdownPreview: React.FC<FilePreviewContentProps> = ({
  content,
}) => {
  return (
    <ScrollArea className="h-full w-full">
      <article className="markdown-body p-6 max-w-4xl mx-auto">
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            // Headings
            h1: ({ children }) => (
              <h1 className="text-3xl font-bold mt-8 mb-4 pb-2 border-b border-border first:mt-0">
                {children}
              </h1>
            ),
            h2: ({ children }) => (
              <h2 className="text-2xl font-semibold mt-6 mb-3 pb-2 border-b border-border">
                {children}
              </h2>
            ),
            h3: ({ children }) => (
              <h3 className="text-xl font-semibold mt-5 mb-2">{children}</h3>
            ),
            h4: ({ children }) => (
              <h4 className="text-lg font-semibold mt-4 mb-2">{children}</h4>
            ),
            h5: ({ children }) => (
              <h5 className="text-base font-semibold mt-3 mb-1">{children}</h5>
            ),
            h6: ({ children }) => (
              <h6 className="text-sm font-semibold mt-3 mb-1 text-muted-foreground">
                {children}
              </h6>
            ),

            // Paragraphs and text
            p: ({ children }) => (
              <p className="my-4 leading-7 text-foreground">{children}</p>
            ),
            strong: ({ children }) => (
              <strong className="font-semibold">{children}</strong>
            ),
            em: ({ children }) => <em className="italic">{children}</em>,
            del: ({ children }) => (
              <del className="line-through text-muted-foreground">{children}</del>
            ),

            // Links
            a: ({ href, children }) => (
              <a
                href={href}
                className="text-primary underline underline-offset-4 hover:text-primary/80 transition-colors"
                target="_blank"
                rel="noopener noreferrer"
              >
                {children}
              </a>
            ),

            // Lists
            ul: ({ children }) => (
              <ul className="my-4 ml-6 list-disc space-y-2">{children}</ul>
            ),
            ol: ({ children }) => (
              <ol className="my-4 ml-6 list-decimal space-y-2">{children}</ol>
            ),
            li: ({ children }) => (
              <li className="leading-7">{children}</li>
            ),

            // Code
            code: ({ className, children, ...props }) => {
              const isInline = !className;
              if (isInline) {
                return (
                  <code
                    className="px-1.5 py-0.5 rounded-md bg-muted text-sm font-mono"
                    {...props}
                  >
                    {children}
                  </code>
                );
              }
              // Block code
              return (
                <code
                  className={`block text-sm font-mono ${className || ''}`}
                  {...props}
                >
                  {children}
                </code>
              );
            },
            pre: ({ children }) => (
              <pre className="my-4 p-4 rounded-lg bg-muted overflow-x-auto">
                {children}
              </pre>
            ),

            // Blockquotes
            blockquote: ({ children }) => (
              <blockquote className="my-4 pl-4 border-l-4 border-primary/50 italic text-muted-foreground">
                {children}
              </blockquote>
            ),

            // Horizontal rule
            hr: () => <hr className="my-8 border-border" />,

            // Tables
            table: ({ children }) => (
              <div className="my-4 overflow-x-auto">
                <table className="w-full border-collapse border border-border">
                  {children}
                </table>
              </div>
            ),
            thead: ({ children }) => (
              <thead className="bg-muted">{children}</thead>
            ),
            tbody: ({ children }) => <tbody>{children}</tbody>,
            tr: ({ children }) => (
              <tr className="border-b border-border">{children}</tr>
            ),
            th: ({ children }) => (
              <th className="px-4 py-2 text-left font-semibold border border-border">
                {children}
              </th>
            ),
            td: ({ children }) => (
              <td className="px-4 py-2 border border-border">{children}</td>
            ),

            // Images
            img: ({ src, alt }) => (
              <img
                src={src}
                alt={alt || ''}
                className="max-w-full h-auto rounded-lg my-4"
              />
            ),

            // Task lists (GFM)
            input: ({ type, checked, disabled }) => {
              if (type === 'checkbox') {
                return (
                  <input
                    type="checkbox"
                    checked={checked}
                    disabled={disabled}
                    className="mr-2 h-4 w-4 rounded border-border"
                    readOnly
                  />
                );
              }
              return <input type={type} />;
            },
          }}
        >
          {content}
        </ReactMarkdown>
      </article>
    </ScrollArea>
  );
};

export default MarkdownPreview;
