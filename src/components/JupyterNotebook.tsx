import React from 'react';
import { Play, Square } from 'lucide-react';

interface NotebookCell {
  type: 'code' | 'markdown';
  content: string;
  output?: string;
  executionCount?: number;
}

interface JupyterNotebookProps {
  title: string;
  cells: NotebookCell[];
}

const JupyterNotebook: React.FC<JupyterNotebookProps> = ({ title, cells }) => {
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-sm overflow-hidden">
      {/* Notebook Header */}
      <div className="bg-gray-50 border-b border-gray-200 px-4 py-3">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-medium text-gray-900">{title}</h3>
          <div className="flex items-center space-x-2 text-sm text-gray-500">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 bg-red-500 rounded-full"></div>
              <div className="w-2 h-2 bg-yellow-500 rounded-full"></div>
              <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            </div>
            <span>Jupyter Notebook</span>
          </div>
        </div>
      </div>

      {/* Notebook Cells */}
      <div className="divide-y divide-gray-100">
        {cells.map((cell, index) => (
          <div key={index} className="flex">
            {/* Cell Input Area */}
            <div className="flex-1">
              {cell.type === 'code' && (
                <>
                  {/* Code Input */}
                  <div className="flex">
                    <div className="w-16 bg-gray-50 border-r border-gray-200 flex items-start justify-center pt-3 text-xs text-gray-500">
                      In [{cell.executionCount || ' '}]:
                    </div>
                    <div className="flex-1 p-3">
                      <div className="relative">
                        <pre className="bg-gray-50 border border-gray-200 rounded p-3 text-sm font-mono overflow-x-auto">
                          <code className="text-gray-800">{cell.content}</code>
                        </pre>
                        <div className="absolute top-2 right-2 flex space-x-1">
                          <button className="p-1 text-gray-400 hover:text-gray-600">
                            <Play className="w-3 h-3" />
                          </button>
                          <button className="p-1 text-gray-400 hover:text-gray-600">
                            <Square className="w-3 h-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Code Output */}
                  {cell.output && (
                    <div className="flex">
                      <div className="w-16 bg-gray-50 border-r border-gray-200 flex items-start justify-center pt-3 text-xs text-gray-500">
                        Out[{cell.executionCount || ' '}]:
                      </div>
                      <div className="flex-1 p-3 bg-white">
                        <pre className="text-sm font-mono text-gray-800 whitespace-pre-wrap">
                          {cell.output}
                        </pre>
                      </div>
                    </div>
                  )}
                </>
              )}

              {cell.type === 'markdown' && (
                <div className="p-4">
                  <div className="prose prose-sm max-w-none">
                    <div dangerouslySetInnerHTML={{ __html: cell.content }} />
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default JupyterNotebook;