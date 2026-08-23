export function chartPalette(isDark) {
  return {
    text: isDark ? 'rgba(220, 228, 244, 0.82)' : 'rgba(40, 44, 52, 0.9)',
    subtext: isDark ? 'rgba(150, 160, 180, 0.55)' : 'rgba(100, 108, 120, 0.65)',
    splitLine: isDark ? 'rgba(128, 138, 157, 0.16)' : 'rgba(60, 66, 80, 0.12)',
    tooltipBg: isDark ? 'rgba(26, 30, 42, 0.94)' : 'rgba(255, 255, 255, 0.96)',
    tooltipBorder: isDark ? 'rgba(255, 255, 255, 0.12)' : 'rgba(0, 0, 0, 0.08)',
    tooltipText: isDark ? '#e6eaf5' : '#333333',
  }
}
