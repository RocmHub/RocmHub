import React, { createContext, useContext, useState, useCallback } from 'react';
import { CheckCircle2, XCircle, AlertTriangle, Info, X } from 'lucide-react';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

interface ToastItem {
  id: string;
  type: ToastType;
  message: string;
}

interface ToastContextValue {
  success: (msg: string) => void;
  error: (msg: string) => void;
  warning: (msg: string) => void;
  info: (msg: string) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export const useToast = (): ToastContextValue => {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be inside ToastProvider');
  return ctx;
};

const ICONS = { success: CheckCircle2, error: XCircle, warning: AlertTriangle, info: Info };
const COLORS = {
  success: { bg: 'bg-emerald-900/90 border-emerald-600/40', icon: 'text-emerald-400', text: 'text-emerald-50' },
  error:   { bg: 'bg-red-950/90 border-red-600/40',     icon: 'text-red-400',     text: 'text-red-50'     },
  warning: { bg: 'bg-amber-950/90 border-amber-600/40', icon: 'text-amber-400',   text: 'text-amber-50'   },
  info:    { bg: 'bg-zinc-900/90 border-zinc-600/40',   icon: 'text-zinc-300',    text: 'text-zinc-50'    },
};

const DURATIONS: Record<ToastType, number> = { success: 4000, error: 7000, warning: 5000, info: 4000 };

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  const remove = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const push = useCallback((msg: string, type: ToastType) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setToasts(prev => [...prev.slice(-4), { id, type, message: msg }]);
    setTimeout(() => remove(id), DURATIONS[type]);
  }, [remove]);

  const value: ToastContextValue = {
    success: (msg) => push(msg, 'success'),
    error:   (msg) => push(msg, 'error'),
    warning: (msg) => push(msg, 'warning'),
    info:    (msg) => push(msg, 'info'),
  };

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div className="fixed bottom-6 right-6 z-[9999] flex flex-col gap-2 pointer-events-none" role="region" aria-label="Notifications">
        {toasts.map(t => {
          const Icon = ICONS[t.type];
          const c = COLORS[t.type];
          return (
            <div
              key={t.id}
              className={`pointer-events-auto animate-slide-in-right flex items-start gap-3 px-4 py-3.5 rounded-xl border backdrop-blur-xl shadow-2xl max-w-[380px] ${c.bg}`}
            >
              <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${c.icon}`} />
              <p className={`text-sm leading-relaxed flex-1 ${c.text}`}>{t.message}</p>
              <button
                onClick={() => remove(t.id)}
                className={`shrink-0 opacity-50 hover:opacity-100 transition-opacity ${c.text}`}
                aria-label="Dismiss"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
};
