import React from 'react';
import Editor, { type OnMount } from '@monaco-editor/react';
import { Loader } from 'lucide-react';

interface CodeEditorProps {
  value: string;
  language: string;
  readOnly?: boolean;
  onChange?: (value: string | undefined) => void;
  className?: string;
}

export const CodeEditor: React.FC<CodeEditorProps> = ({
  value,
  language,
  readOnly = false,
  onChange,
  className,
}) => {
  const handleEditorDidMount: OnMount = (editor) => {
    // Configure editor settings
    editor.updateOptions({
      minimap: { enabled: false },
      scrollBeyondLastLine: false,
      fontSize: 14,
      fontFamily: "'JetBrains Mono', 'Fira Code', Consolas, monospace",
      readOnly: readOnly,
      domReadOnly: readOnly,
    });
  };

  return (
    <div className={className || "h-full w-full"}>
      <Editor
        height="100%"
        defaultLanguage={language}
        language={language}
        value={value}
        onChange={onChange}
        onMount={handleEditorDidMount}
        theme="vs-light" // You might want to detect system theme here
        loading={
          <div className="flex items-center justify-center h-full">
            <Loader className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        }
        options={{
          readOnly: readOnly,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          fontSize: 14,
          automaticLayout: true,
        }}
      />
    </div>
  );
};
