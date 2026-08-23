import { defineStore } from 'pinia'

const THEME_KEY = 'gre-theme'

function loadTheme() {
  const saved = localStorage.getItem(THEME_KEY)
  if (saved === 'light' || saved === 'dark') return saved
  return window.matchMedia?.('(prefers-color-scheme: light)').matches ? 'light' : 'dark'
}

export const useAppStore = defineStore('app', {
  state: () => ({
    theme: loadTheme(),
    runs: [],
    latestRun: null,
    selectedRun: null,
  }),
  getters: {
    isDark: (s) => s.theme === 'dark',
  },
  actions: {
    toggleTheme() {
      this.theme = this.theme === 'dark' ? 'light' : 'dark'
      localStorage.setItem(THEME_KEY, this.theme)
    },
    setRuns(runs, latest) {
      this.runs = runs || []
      this.latestRun = latest || this.runs[0] || null
      if (!this.selectedRun) this.selectedRun = this.latestRun
    },
    setSelectedRun(run) {
      this.selectedRun = run
    },
  },
})
