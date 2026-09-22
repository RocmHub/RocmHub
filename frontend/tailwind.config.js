/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: '#101014',
        surface: {
          DEFAULT: '#19191F',
          elevated: '#222229',
          border: '#2E2E38',
        },
        accent: {
          red: '#ED1C24',
          'red-hover': '#D0171E',
          'red-muted': 'rgba(237, 28, 36, 0.15)',
        },
        content: {
          primary: '#F5F5F7',
          secondary: '#A1A1AA',
          muted: '#71717A',
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
    },
  },
  plugins: [],
}
