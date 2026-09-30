import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';
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
  success: { bg: 'bg-emerald-50 border-emerald-200', icon: 'text-emerald-700', text: 'text-emerald-900' },
  error:   { bg: 'bg-red-50 border-red-200', icon: 'text-red-700', text: 'text-red-900' },
  warning: { bg: 'bg-amber-50 border-amber-200', icon: 'text-amber-700', text: 'text-amber-900' },
  info:    { bg: 'bg-zinc-50 border-zinc-200', icon: 'text-zinc-700', text: 'text-zinc-900' },
};

const DURATIONS: Record<ToastType, number> = { success: 4000, error: 7000, warning: 5000, info: 4000 };

export const ToastProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    const clearOnNavigation = () => setToasts([]);
    window.addEventListener('hashchange', clearOnNavigation);
    return () => window.removeEventListener('hashchange', clearOnNavigation);
  }, []);

  const remove = useCallback((id: string) => {
    setToasts(prev => prev.filter(t => t.id !== id));
  }, []);

  const push = useCallback((msg: string, type: ToastType) => {
    const id = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    setToasts([{ id, type, message: msg }]);
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
              data-toast-type={t.type}
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
