<script setup>
import { computed } from 'vue'
import VChart from '@/plugins/echarts'
import { RATING_ORDER, RATING_COLORS } from '@/utils/rating'
import { chartPalette } from '@/utils/chartTheme'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  items: { type: Array, default: () => [] },
})

const store = useAppStore()

const option = computed(() => {
  const pal = chartPalette(store.isDark)
  const counts = {}
  for (const it of props.items) {
    const r = it.actual_rating
    if (r) counts[r] = (counts[r] || 0) + 1
  }
  const data = RATING_ORDER.filter((r) => counts[r]).map((r) => ({
    name: r,
    value: counts[r],
    itemStyle: { color: RATING_COLORS[r] },
  }))
  return {
    color: RATING_ORDER.map((r) => RATING_COLORS[r]),
    tooltip: {
      trigger: 'item',
      backgroundColor: pal.tooltipBg,
      borderColor: pal.tooltipBorder,
      textStyle: { color: pal.tooltipText },
      formatter: '{b}: {c} ({d}%)',
    },
    legend: { orient: 'vertical', right: 8, top: 'middle', textStyle: { color: pal.text } },
    series: [
      {
        type: 'pie',
        radius: ['55%', '80%'],
        center: ['42%', '50%'],
        avoidLabelOverlap: true,
        itemStyle: { borderColor: pal.tooltipBg, borderWidth: 2 },
        label: { show: false },
        data,
      },
    ],
  }
})
</script>

<template>
  <v-chart class="chart" :option="option" autoresize />
</template>

<style scoped>
.chart {
  height: 260px;
}
</style>
