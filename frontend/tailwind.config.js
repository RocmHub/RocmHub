/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#0D0D11',
        surface: {
          DEFAULT: '#16161C',
          elevated: '#1E1E26',
          card: '#1A1A22',
          deep: '#111116',
          border: '#2A2A35',
        },
        accent: {
          red: '#ED1C24',
          'red-hover': '#D0171E',
          'red-muted': 'rgba(237, 28, 36, 0.12)',
          'red-glow': 'rgba(237, 28, 36, 0.20)',
          subtle: 'rgba(237, 28, 36, 0.06)',
        },
        content: {
          primary: '#F2F2F5',
          secondary: '#9898A8',
          muted: '#60606E',
          inverse: '#0D0D11',
        },
        status: {
          success: '#10B981',
          warning: '#F59E0B',
          error: '#EF4444',
          info: '#3B82F6',
          purple: '#8B5CF6',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'Menlo', 'Monaco', 'Courier New', 'monospace'],
      },
      keyframes: {
        shimmer: {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'fade-up': {
          '0%': { opacity: '0', transform: 'translateY(8px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        'slide-in-right': {
          '0%': { opacity: '0', transform: 'translateX(16px)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
      },
      animation: {
        shimmer: 'shimmer 2.5s linear infinite',
        'fade-up': 'fade-up 0.3s ease-out',
        'slide-in-right': 'slide-in-right 0.3s ease-out',
        'pulse-slow': 'pulse 3s ease-in-out infinite',
      },
    },
  },
  plugins: [],
}
