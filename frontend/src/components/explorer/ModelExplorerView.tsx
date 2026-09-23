import React, { useState } from 'react';
import type { ModelMetadata } from '../../api/types';
import { fetchModel, ApiError } from '../../api/client';
import { Search, CheckCircle2, XCircle, AlertTriangle, ArrowRight, Shield } from 'lucide-react';
import type { NavTab } from '../layout/Sidebar';

interface ModelExplorerViewProps {
  onSelectModelForForge: (modelId: string, revision?: string) => void;
  onNavigate: (tab: NavTab) => void;
}

export const ModelExplorerView: React.FC<ModelExplorerViewProps> = ({
  onSelectModelForForge,
}) => {
  const [modelIdInput, setModelIdInput] = useState('Qwen/Qwen2.5-0.5B-Instruct');
  const [revisionInput, setRevisionInput] = useState('main');
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInspect = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!modelIdInput.trim()) return;

    setIsLoading(true);
    setError(null);
    try {
      const data = await fetchModel(modelIdInput.trim(), revisionInput.trim() || 'main');
      setMetadata(data);
    } catch (err: any) {
      if (err instanceof ApiError) {
        setError(err.message);
      } else {
        setError(err.message || 'Failed to inspect model');
      }
      setMetadata(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-6 max-w-5xl">
      <div>
        <h1 className="text-xl font-bold text-content-primary">Model Explorer</h1>
        <p className="text-xs text-content-secondary mt-0.5">
          Inspect Hugging Face model metadata, resolve immutable commit SHAs, and check AMD compatibility.
        </p>
      </div>

      {/* Input Form */}
      <form onSubmit={handleInspect} className="p-4 rounded-lg bg-surface border border-surface-border space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div className="md:col-span-3 space-y-1">
            <label className="text-xs font-mono text-content-secondary">Hugging Face Model ID</label>
            <div className="relative">
              <input
                type="text"
                value={modelIdInput}
                onChange={(e) => setModelIdInput(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct"
                className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red transition-colors"
              />
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-xs font-mono text-content-secondary">Revision / Commit</label>
            <input
              type="text"
              value={revisionInput}
              onChange={(e) => setRevisionInput(e.target.value)}
              placeholder="main"
              className="w-full bg-background border border-surface-border rounded px-3 py-2 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red transition-colors"
            />
          </div>
        </div>

        <div className="flex items-center justify-between pt-1">
          <div className="flex space-x-2 text-[11px] font-mono text-zinc-500">
            <span>Examples:</span>
            <button
              type="button"
              onClick={() => {
                setModelIdInput('Qwen/Qwen2.5-0.5B-Instruct');
                setRevisionInput('main');
              }}
              className="hover:text-zinc-300 underline"
            >
              Qwen2.5-0.5B-Instruct
            </button>
          </div>
          <button
            type="submit"
            disabled={isLoading}
            className="flex items-center space-x-2 px-4 py-2 rounded bg-accent-red hover:bg-accent-red-hover text-white text-xs font-medium transition-colors disabled:opacity-50"
          >
            <Search className="w-3.5 h-3.5" />
            <span>{isLoading ? 'Resolving...' : 'Inspect Model'}</span>
          </button>
        </div>
      </form>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/20 text-xs text-red-400 flex items-start space-x-2">
          <XCircle className="w-4 h-4 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <span className="font-semibold">Inspection Error</span>
            <p>{error}</p>
          </div>
        </div>
      )}

      {/* Results view */}
      {metadata && (
        <div className="space-y-4">
          {/* Main Specs Card */}
          <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-4">
            <div className="flex items-start justify-between">
              <div>
                <h2 className="text-base font-bold text-content-primary font-mono">{metadata.model_id}</h2>
                <div className="flex items-center space-x-2 text-xs font-mono text-zinc-400 mt-1">
                  <span>Immutable SHA:</span>
                  <span className="text-zinc-200 bg-black/40 px-1.5 py-0.5 rounded border border-zinc-800 break-all">
                    {metadata.commit_sha}
                  </span>
                </div>
              </div>
              <button
                onClick={() => onSelectModelForForge(metadata.model_id, metadata.commit_sha)}
                className="flex items-center space-x-1.5 px-3 py-1.5 rounded bg-accent-red hover:bg-accent-red-hover text-white text-xs font-medium transition-colors"
              >
                <span>Open in Forge Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t border-surface-border text-xs font-mono">
              <div className="space-y-0.5">
                <span className="text-zinc-500">Architecture</span>
                <div className="text-content-primary font-semibold">{metadata.architecture || 'Unknown'}</div>
              </div>
              <div className="space-y-0.5">
                <span className="text-zinc-500">Parameters</span>
                <div className="text-content-primary font-semibold">
                  {metadata.parameter_count
                    ? `${(metadata.parameter_count / 1e9).toFixed(2)}B`
                    : 'Not specified'}
                </div>
              </div>
              <div className="space-y-0.5">
                <span className="text-zinc-500">Context Length</span>
                <div className="text-content-primary font-semibold">
                  {metadata.context_length ? `${metadata.context_length.toLocaleString()} tokens` : 'N/A'}
                </div>
              </div>
              <div className="space-y-0.5">
                <span className="text-zinc-500">Weights Format</span>
                <div className="text-content-primary font-semibold">{metadata.weights_format || 'safetensors'}</div>
              </div>
            </div>

            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 border-t border-surface-border text-xs font-mono">
              <div className="space-y-0.5">
                <span className="text-zinc-500">License</span>
                <div className="text-content-primary font-semibold">{metadata.license || 'Open / Unspecified'}</div>
              </div>
              <div className="space-y-0.5">
                <span className="text-zinc-500">Pipeline Tag</span>
                <div className="text-content-primary font-semibold">{metadata.pipeline_tag || 'text-generation'}</div>
              </div>
              <div className="space-y-0.5">
                <span className="text-zinc-500">Files Count</span>
                <div className="text-content-primary font-semibold">{metadata.files_count} repository files</div>
              </div>
              <div className="space-y-0.5">
                <span className="text-zinc-500">Estimated Disk</span>
                <div className="text-content-primary font-semibold">
                  {metadata.parameter_count
                    ? `~${((metadata.parameter_count * 2 * 1.15) / 1e9).toFixed(2)} GB (FP16)`
                    : '~2 GB'}
                </div>
              </div>
            </div>
          </div>

          {/* Compatibility Assessment Card */}
          <div className="p-5 rounded-lg bg-surface border border-surface-border space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-content-primary flex items-center space-x-2">
                <Shield className="w-4 h-4 text-zinc-400" />
                <span>Host Hardware Compatibility Assessment</span>
              </h3>
              <div>
                {metadata.compatibility.verdict === 'READY' ? (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
                    READY FOR AMD
                  </span>
                ) : metadata.compatibility.verdict === 'NO_ACCELERATOR' ? (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
                    <AlertTriangle className="w-3.5 h-3.5 mr-1" />
                    CONFIG_ONLY (NO AMD GPU)
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2 py-0.5 rounded text-xs font-mono font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/20">
                    {metadata.compatibility.verdict}
                  </span>
                )}
              </div>
            </div>

            {/* Incompatibility / Evaluation Reasons */}
            <div className="space-y-2 pt-2 text-xs font-mono">
              {metadata.compatibility.reasons.length > 0 ? (
                metadata.compatibility.reasons.map((r, i) => (
                  <div
                    key={i}
                    className="p-2.5 rounded bg-surface-elevated border border-surface-border flex items-start space-x-2"
                  >
                    <span className="text-zinc-500">[{r.code}]</span>
                    <span className="text-zinc-300">{r.message}</span>
                  </div>
                ))
              ) : (
                <div className="text-zinc-500">No architectural blockers detected for causal language model.</div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
