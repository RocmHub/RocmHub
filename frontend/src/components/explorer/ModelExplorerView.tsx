import React, { useState } from 'react';
import type { ModelMetadata } from '../../api/types';
import { fetchModel, ApiError } from '../../api/client';
import {
  Search, CheckCircle2, XCircle, AlertTriangle, ArrowRight,
  Shield, Layers, HardDrive, FileCode, Hash,
} from 'lucide-react';
import type { NavTab } from '../layout/Sidebar';
import { useToast } from '../common/Toast';

import modelExplorerHeroImg from '../../assets/visuals/model_explorer_hero.jpg';

interface ModelExplorerViewProps {
  onSelectModelForForge: (modelId: string, revision?: string) => void;
  onNavigate: (tab: NavTab) => void;
}

const PRESET_MODELS = [
  { id: 'Qwen/Qwen2.5-0.5B-Instruct', label: 'Qwen2.5 0.5B' },
  { id: 'Qwen/Qwen2.5-1.5B-Instruct', label: 'Qwen2.5 1.5B' },
  { id: 'Qwen/Qwen2.5-7B-Instruct', label: 'Qwen2.5 7B' },
];

export const ModelExplorerView: React.FC<ModelExplorerViewProps> = ({
  onSelectModelForForge,
}) => {
  const toast = useToast();
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
      const msg = err instanceof ApiError ? err.message : (err.message || 'Failed to inspect model');
      setError(msg);
      toast.error(msg);
      setMetadata(null);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="page-fade h-full flex flex-col">
      {/* ── SEARCH HERO ──────────────────────────────────────────────── */}
      <div className="relative overflow-hidden border-b border-surface-border">
        <div
          className="absolute inset-0 bg-cover bg-center opacity-40"
          style={{ backgroundImage: `url(${modelExplorerHeroImg})` }}
        />
        <div className="absolute inset-0 hero-overlay" />

        <div className="relative z-10 px-8 sm:px-12 py-10">
          <div className="max-w-2xl">
            <p className="text-xs font-mono font-medium text-blue-400/90 uppercase tracking-wider mb-3">
              Model Discovery
            </p>
            <h1 className="text-2xl font-bold tracking-tight text-white mb-1.5">Model Explorer</h1>
            <p className="text-sm text-content-secondary mb-7">
              Find and validate AI models for AMD hardware.
            </p>

            <form onSubmit={handleInspect} className="space-y-3">
              <div className="flex gap-3">
                <div className="flex-1 relative">
                  <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-content-muted pointer-events-none" />
                  <input
                    type="text"
                    value={modelIdInput}
                    onChange={(e) => setModelIdInput(e.target.value)}
                    placeholder="org/model-name (e.g. Qwen/Qwen2.5-0.5B-Instruct)"
                    className="w-full bg-background/80 backdrop-blur border border-surface-border rounded-lg pl-10 pr-4 py-3 text-sm text-content-primary placeholder:text-content-muted focus:outline-none focus:border-accent-red focus:ring-1 focus:ring-accent-red/20 transition-all"
                  />
                </div>
                <button
                  type="submit"
                  disabled={isLoading}
                  className="btn-primary shrink-0"
                >
                  {isLoading ? (
                    <><span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Inspecting...</>
                  ) : (
                    <><Search className="w-4 h-4" /> Inspect</>
                  )}
                </button>
              </div>

              {/* Preset chips */}
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[11px] text-content-muted font-mono">Popular:</span>
                {PRESET_MODELS.map((p) => (
                  <button
                    key={p.id}
                    type="button"
                    onClick={() => { setModelIdInput(p.id); setRevisionInput('main'); }}
                    className={`px-2.5 py-1 rounded-full text-[11px] border transition-all ${
                      modelIdInput === p.id
                        ? 'bg-accent-red/15 border-accent-red/40 text-red-300'
                        : 'bg-black/30 border-surface-border text-content-secondary hover:text-content-primary hover:border-zinc-600'
                    }`}
                  >
                    {p.label}
                  </button>
                ))}
                <div className="flex items-center gap-1.5 ml-auto">
                  <span className="text-[11px] text-content-muted font-mono">Revision:</span>
                  <input
                    type="text"
                    value={revisionInput}
                    onChange={(e) => setRevisionInput(e.target.value)}
                    placeholder="main"
                    className="w-20 bg-black/30 backdrop-blur border border-surface-border rounded px-2.5 py-1 text-[11px] font-mono text-content-primary focus:outline-none focus:border-accent-red transition-all"
                  />
                </div>
              </div>
            </form>
          </div>
        </div>
      </div>

      {/* ── RESULTS ──────────────────────────────────────────────────── */}
      <div className="flex-1 overflow-y-auto">
        {/* Error */}
        {error && (
          <div className="mx-8 mt-6 p-4 rounded-xl bg-red-950/50 border border-red-500/25 flex items-start gap-3">
            <XCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <div className="text-sm font-semibold text-red-300 mb-1">Inspection Failed</div>
              <p className="text-xs text-red-400/80 font-mono leading-relaxed">{error}</p>
            </div>
          </div>
        )}

        {/* Empty state */}
        {!metadata && !isLoading && !error && (
          <div className="px-8 py-16 text-center">
            <div className="w-14 h-14 rounded-2xl bg-surface-elevated border border-surface-border flex items-center justify-center mx-auto mb-4">
              <Search className="w-6 h-6 text-content-muted" />
            </div>
            <div className="text-sm font-semibold text-content-secondary mb-1">Ready to inspect</div>
            <p className="text-xs text-content-muted max-w-sm mx-auto leading-relaxed">
              Enter any public Hugging Face model ID to resolve its immutable commit SHA, check architecture compatibility, and prepare it for AMD GPU builds.
            </p>
          </div>
        )}

        {/* Result: split layout */}
        {metadata && (
          <div className="px-8 py-6">
            <div className="grid grid-cols-1 xl:grid-cols-[1fr_340px] gap-5 items-start">

              {/* Left: Model details */}
              <div className="space-y-4">
                {/* Main card */}
                <div className="card p-6 space-y-5">
                  {/* Header */}
                  <div className="flex flex-col sm:flex-row sm:items-start gap-4 justify-between">
                    <div className="space-y-2 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-xs font-mono text-content-muted">Resolved</span>
                        <span className="w-1 h-1 rounded-full bg-surface-border" />
                        <span className="text-xs font-mono text-content-muted">{metadata.pipeline_tag || 'text-generation'}</span>
                      </div>
                      <h2 className="text-xl font-bold text-content-primary font-mono break-all">{metadata.model_id}</h2>
                    </div>
                    <button
                      onClick={() => onSelectModelForForge(metadata.model_id, metadata.commit_sha)}
                      className="btn-primary shrink-0 self-start"
                    >
                      Open in Forge Studio
                      <ArrowRight className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Immutable SHA */}
                  <div className="flex items-start gap-3 p-3 rounded-lg bg-surface-elevated border border-surface-border">
                    <Hash className="w-4 h-4 text-content-muted shrink-0 mt-0.5" />
                    <div className="min-w-0">
                      <div className="text-[11px] text-content-muted font-medium mb-1">Immutable SHA</div>
                      <div className="text-xs font-mono text-content-primary break-all">{metadata.commit_sha}</div>
                    </div>
                  </div>

                  {/* Spec grid row 1 */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-surface-border">
                    <div>
                      <div className="flex items-center gap-1 text-[11px] text-content-muted mb-1">
                        <Layers className="w-3 h-3" /> Architecture
                      </div>
                      <div className="text-sm font-semibold text-content-primary font-mono">{metadata.architecture || '—'}</div>
                    </div>
                    <div>
                      <div className="text-[11px] text-content-muted mb-1">Parameters</div>
                      <div className="text-sm font-semibold text-content-primary font-mono">
                        {metadata.parameter_count ? `${(metadata.parameter_count / 1e9).toFixed(2)}B` : '—'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] text-content-muted mb-1">Context</div>
                      <div className="text-sm font-semibold text-content-primary font-mono">
                        {metadata.context_length ? `${metadata.context_length.toLocaleString()} tok` : '—'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] text-content-muted mb-1">Format</div>
                      <div className="text-sm font-semibold text-content-primary font-mono">{metadata.weights_format || 'safetensors'}</div>
                    </div>
                  </div>

                  {/* Spec grid row 2 */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-4 border-t border-surface-border">
                    <div>
                      <div className="flex items-center gap-1 text-[11px] text-content-muted mb-1">
                        <FileCode className="w-3 h-3" /> License
                      </div>
                      <div className="text-sm font-semibold text-content-primary">{metadata.license || 'Open'}</div>
                    </div>
                    <div>
                      <div className="text-[11px] text-content-muted mb-1">Files</div>
                      <div className="text-sm font-semibold text-content-primary">{metadata.files_count} files</div>
                    </div>
                    <div>
                      <div className="flex items-center gap-1 text-[11px] text-content-muted mb-1">
                        <HardDrive className="w-3 h-3" /> Est. Memory (FP16)
                      </div>
                      <div className="text-sm font-semibold text-content-primary font-mono">
                        {metadata.parameter_count
                          ? `~${((metadata.parameter_count * 2 * 1.15) / 1e9).toFixed(2)} GB`
                          : '~2 GB'}
                      </div>
                    </div>
                    <div>
                      <div className="text-[11px] text-content-muted mb-1">Pipeline</div>
                      <div className="text-sm font-semibold text-content-primary">{metadata.pipeline_tag || 'text-generation'}</div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right: Compatibility panel */}
              <div className="card p-5 space-y-4">
                <div className="flex items-center gap-2 mb-1">
                  <Shield className="w-4 h-4 text-content-muted" />
                  <h3 className="text-sm font-semibold text-content-primary">AMD Compatibility</h3>
                </div>

                {/* Verdict badge */}
                <div>
                  {metadata.compatibility.verdict === 'READY' ? (
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/20">
                      <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0" />
                      <div>
                        <div className="text-xs font-semibold text-emerald-300">Ready for AMD ROCm</div>
                        <div className="text-[11px] text-emerald-400/70 mt-0.5">Architecture validated</div>
                      </div>
                    </div>
                  ) : metadata.compatibility.verdict === 'NO_ACCELERATOR' ? (
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-amber-500/10 border border-amber-500/20">
                      <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
                      <div>
                        <div className="text-xs font-semibold text-amber-300">CONFIG ONLY</div>
                        <div className="text-[11px] text-amber-400/70 mt-0.5">No AMD GPU on this host</div>
                      </div>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 p-3 rounded-lg bg-surface-elevated border border-surface-border">
                      <Shield className="w-4 h-4 text-content-muted shrink-0" />
                      <div className="text-xs font-mono text-content-secondary">{metadata.compatibility.verdict}</div>
                    </div>
                  )}
                </div>

                {/* Reasons */}
                <div className="space-y-2">
                  {metadata.compatibility.reasons.length > 0 ? (
                    metadata.compatibility.reasons.map((r, i) => (
                      <div key={i} className="p-2.5 rounded-lg bg-surface-elevated border border-surface-border text-xs">
                        <span className="font-mono text-content-muted font-semibold">[{r.code}]</span>{' '}
                        <span className="text-content-secondary">{r.message}</span>
                      </div>
                    ))
                  ) : (
                    <div className="flex items-start gap-2 p-2.5 rounded-lg bg-emerald-500/5 border border-emerald-500/15 text-xs text-emerald-300">
                      <CheckCircle2 className="w-3.5 h-3.5 shrink-0 mt-0.5 text-emerald-400" />
                      <span>Causal LM architecture validated for standard ROCm preparation.</span>
                    </div>
                  )}
                </div>

                {/* CTA */}
                <button
                  onClick={() => onSelectModelForForge(metadata.model_id, metadata.commit_sha)}
                  className="w-full btn-primary justify-center"
                >
                  Build for AMD
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
