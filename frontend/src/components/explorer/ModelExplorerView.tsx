import React, { useState } from 'react';
import type { ModelMetadata } from '../../api/types';
import { fetchModel, ApiError } from '../../api/client';
import { Search, CheckCircle2, XCircle, AlertTriangle, ArrowRight, Shield, Layers, HardDrive, FileCode, Sparkles } from 'lucide-react';
import type { NavTab } from '../layout/Sidebar';

import modelExplorerHeroImg from '../../assets/visuals/model_explorer_hero.jpg';
import emptySearchImg from '../../assets/visuals/empty_search.svg';

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
      {/* Visual Header Banner */}
      <div className="relative rounded-xl overflow-hidden border border-surface-border bg-surface shadow-xl">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-30 mix-blend-luminosity"
          style={{ backgroundImage: `url(${modelExplorerHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay" />

        <div className="relative p-6 space-y-2">
          <div className="inline-flex items-center space-x-2 px-2.5 py-1 rounded-full bg-blue-500/10 border border-blue-500/30 text-[11px] font-mono text-blue-400">
            <Layers className="w-3.5 h-3.5 text-blue-400" />
            <span>Architecture &amp; Checksum Inspection</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-white">Model Explorer</h1>
          <p className="text-xs text-zinc-300 max-w-2xl font-sans leading-relaxed">
            Inspect Hugging Face model metadata, resolve immutable commit SHAs, and check AMD compatibility.
          </p>
        </div>
      </div>

      {/* Input Form Card */}
      <form onSubmit={handleInspect} className="p-5 rounded-xl bg-surface border border-surface-border shadow-sm space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3.5">
          <div className="md:col-span-3 space-y-1.5">
            <label className="text-xs font-mono text-content-secondary flex items-center space-x-1.5">
              <span>Hugging Face Model ID</span>
              <span className="text-[10px] text-zinc-500 font-sans">(Repository format: org/model-name)</span>
            </label>
            <div className="relative">
              <input
                type="text"
                value={modelIdInput}
                onChange={(e) => setModelIdInput(e.target.value)}
                placeholder="e.g. Qwen/Qwen2.5-0.5B-Instruct"
                className="w-full bg-background border border-surface-border rounded-lg px-3.5 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
              />
            </div>
          </div>
          <div className="space-y-1.5">
            <label className="text-xs font-mono text-content-secondary">Revision / Branch</label>
            <input
              type="text"
              value={revisionInput}
              onChange={(e) => setRevisionInput(e.target.value)}
              placeholder="main"
              className="w-full bg-background border border-surface-border rounded-lg px-3.5 py-2.5 text-xs font-mono text-content-primary focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red transition-all"
            />
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-surface-border/60">
          <div className="flex items-center space-x-2 text-[11px] font-mono text-zinc-400">
            <span>Quick Samples:</span>
            <button
              type="button"
              onClick={() => {
                setModelIdInput('Qwen/Qwen2.5-0.5B-Instruct');
                setRevisionInput('main');
              }}
              className="px-2 py-0.5 rounded bg-surface-elevated hover:bg-zinc-700 hover:text-white border border-surface-border text-zinc-300 transition-colors"
            >
              Qwen2.5-0.5B
            </button>
            <button
              type="button"
              onClick={() => {
                setModelIdInput('Qwen/Qwen2.5-1.5B-Instruct');
                setRevisionInput('main');
              }}
              className="px-2 py-0.5 rounded bg-surface-elevated hover:bg-zinc-700 hover:text-white border border-surface-border text-zinc-300 transition-colors"
            >
              Qwen2.5-1.5B
            </button>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-accent-red hover:bg-accent-red-hover text-white text-xs font-semibold shadow-md shadow-red-950/30 transition-all hover:scale-[1.02] active:scale-[0.98] disabled:opacity-50"
          >
            <Search className="w-3.5 h-3.5" />
            <span>{isLoading ? 'Resolving Metadata...' : 'Inspect Model'}</span>
          </button>
        </div>
      </form>

      {/* Error state */}
      {error && (
        <div className="p-4 rounded-xl bg-red-500/10 border border-red-500/25 text-xs text-red-400 flex items-start space-x-3 shadow-sm">
          <XCircle className="w-5 h-5 shrink-0 mt-0.5 text-red-400" />
          <div className="space-y-1">
            <span className="font-semibold text-red-300">Model Resolution Failure</span>
            <p className="font-mono text-[11px] text-red-300/90 leading-relaxed">{error}</p>
          </div>
        </div>
      )}

      {/* Empty State before any inspection */}
      {!metadata && !isLoading && !error && (
        <div className="p-10 rounded-xl bg-surface border border-surface-border text-center space-y-4">
          <img src={emptySearchImg} alt="Search Model" className="w-40 h-28 mx-auto opacity-70" />
          <div className="space-y-1 max-w-md mx-auto">
            <div className="text-xs font-mono font-semibold text-zinc-300">Ready to Inspect Models</div>
            <p className="text-[11px] text-zinc-500 leading-relaxed">
              Enter any public Hugging Face model repository identifier above. ROCmHub will query Hugging Face API to resolve immutable commit SHAs, tensor requirements, and architectural compatibility.
            </p>
          </div>
        </div>
      )}

      {/* Results view */}
      {metadata && (
        <div className="space-y-5">
          {/* Main Specs Card */}
          <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-lg space-y-5">
            <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
              <div className="space-y-1.5">
                <div className="inline-flex items-center space-x-1.5 px-2 py-0.5 rounded text-[10px] font-mono bg-surface-elevated text-zinc-400 border border-surface-border">
                  <Sparkles className="w-3 h-3 text-accent-red" />
                  <span>Resolved Model Spec</span>
                </div>
                <h2 className="text-lg font-bold text-content-primary font-mono">{metadata.model_id}</h2>
                <div className="flex flex-wrap items-center gap-2 text-xs font-mono text-zinc-400">
                  <span className="text-zinc-500">Immutable SHA:</span>
                  <span className="text-zinc-200 bg-black/60 px-2 py-0.5 rounded border border-zinc-800 break-all text-[11px]">
                    {metadata.commit_sha}
                  </span>
                </div>
              </div>

              <button
                onClick={() => onSelectModelForForge(metadata.model_id, metadata.commit_sha)}
                className="inline-flex items-center space-x-2 px-4 py-2 rounded-lg bg-accent-red hover:bg-accent-red-hover text-white text-xs font-semibold shadow-md shadow-red-950/40 transition-all hover:scale-[1.02] shrink-0 self-start"
              >
                <span>Open in Forge Studio</span>
                <ArrowRight className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Architecture Grid Row 1 */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-3 border-t border-surface-border text-xs font-mono">
              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px] flex items-center space-x-1">
                  <Layers className="w-3 h-3" />
                  <span>Architecture</span>
                </span>
                <div className="text-content-primary font-semibold text-sm">{metadata.architecture || 'Unknown'}</div>
              </div>

              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px]">Parameters</span>
                <div className="text-content-primary font-semibold text-sm">
                  {metadata.parameter_count
                    ? `${(metadata.parameter_count / 1e9).toFixed(2)}B`
                    : 'Not specified'}
                </div>
              </div>

              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px]">Context Length</span>
                <div className="text-content-primary font-semibold text-sm">
                  {metadata.context_length ? `${metadata.context_length.toLocaleString()} tokens` : 'N/A'}
                </div>
              </div>

              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px]">Weights Format</span>
                <div className="text-content-primary font-semibold text-sm">{metadata.weights_format || 'safetensors'}</div>
              </div>
            </div>

            {/* Architecture Grid Row 2 */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-3 border-t border-surface-border text-xs font-mono">
              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px] flex items-center space-x-1">
                  <FileCode className="w-3 h-3" />
                  <span>License</span>
                </span>
                <div className="text-content-primary font-semibold">{metadata.license || 'Open / Unspecified'}</div>
              </div>

              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px]">Pipeline Tag</span>
                <div className="text-content-primary font-semibold">{metadata.pipeline_tag || 'text-generation'}</div>
              </div>

              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px]">Repository Files</span>
                <div className="text-content-primary font-semibold">{metadata.files_count} files</div>
              </div>

              <div className="space-y-1">
                <span className="text-zinc-500 text-[11px] flex items-center space-x-1">
                  <HardDrive className="w-3 h-3" />
                  <span>Est. Memory (FP16)</span>
                </span>
                <div className="text-content-primary font-semibold">
                  {metadata.parameter_count
                    ? `~${((metadata.parameter_count * 2 * 1.15) / 1e9).toFixed(2)} GB`
                    : '~2 GB'}
                </div>
              </div>
            </div>
          </div>

          {/* Compatibility Assessment Card */}
          <div className="p-6 rounded-xl bg-surface border border-surface-border shadow-sm space-y-3.5">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-content-primary flex items-center space-x-2">
                <Shield className="w-4 h-4 text-zinc-400" />
                <span>Preflight Hardware Compatibility Assessment</span>
              </h3>
              <div>
                {metadata.compatibility.verdict === 'READY' ? (
                  <span className="inline-flex items-center px-2.5 py-1 rounded text-xs font-mono font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/25">
                    <CheckCircle2 className="w-3.5 h-3.5 mr-1.5" />
                    READY FOR AMD ROCM
                  </span>
                ) : metadata.compatibility.verdict === 'NO_ACCELERATOR' ? (
                  <span className="inline-flex items-center px-2.5 py-1 rounded text-xs font-mono font-medium bg-amber-500/10 text-amber-400 border border-amber-500/25">
                    <AlertTriangle className="w-3.5 h-3.5 mr-1.5" />
                    CONFIG_ONLY (NO AMD ACCELERATOR)
                  </span>
                ) : (
                  <span className="inline-flex items-center px-2.5 py-1 rounded text-xs font-mono font-medium bg-zinc-500/10 text-zinc-400 border border-zinc-500/25">
                    {metadata.compatibility.verdict}
                  </span>
                )}
              </div>
            </div>

            {/* Incompatibility / Evaluation Reasons */}
            <div className="space-y-2 pt-1 text-xs font-mono">
              {metadata.compatibility.reasons.length > 0 ? (
                metadata.compatibility.reasons.map((r, i) => (
                  <div
                    key={i}
                    className="p-3 rounded-lg bg-surface-elevated border border-surface-border flex items-start space-x-2.5"
                  >
                    <span className="text-zinc-500 font-semibold">[{r.code}]</span>
                    <span className="text-zinc-300">{r.message}</span>
                  </div>
                ))
              ) : (
                <div className="p-3 rounded-lg bg-emerald-500/5 border border-emerald-500/20 text-emerald-300 text-xs">
                  &#x2713; Causal language model architecture is validated for standard ROCm preparation recipes.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
