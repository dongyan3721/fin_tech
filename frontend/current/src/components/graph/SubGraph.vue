<script setup>
import { computed } from 'vue'
import VChart from '@/plugins/echarts'
import { RATING_COLORS } from '@/utils/rating'
import { chartPalette } from '@/utils/chartTheme'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  center: { type: String, default: '' },
  nodes: { type: Array, default: () => [] }, // [{ id, rating }]
  links: { type: Array, default: () => [] }, // [{ source, target, relationship }]
})

const emit = defineEmits(['node-click'])

const store = useAppStore()

const option = computed(() => {
  const pal = chartPalette(store.isDark)
  return {
    tooltip: {
      trigger: 'item',
      backgroundColor: pal.tooltipBg,
      borderColor: pal.tooltipBorder,
      textStyle: { color: pal.tooltipText },
      formatter: (p) => (p.dataType === 'edge'
        ? `${p.data.source} → ${p.data.target}`
        : `${p.data.name}<br/>评级: ${p.data.rating || '—'}`),
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        roam: true,
        draggable: true,
        data: props.nodes.map((n) => ({
          id: n.id,
          name: n.id,
          rating: n.rating,
          symbolSize: n.id === props.center ? 46 : 26,
          itemStyle: { color: RATING_COLORS[n.rating] || '#8a9099' },
        })),
        links: props.links.map((l) => ({
          source: l.source,
          target: l.target,
          lineStyle: { color: l.relationship === 'supply' ? '#4c8dff' : '#36ad6a', opacity: 0.5, width: 1.5 },
        })),
        label: { show: true, position: 'right', fontSize: 10, color: pal.text, formatter: '{b}' },
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: 6,
        force: { repulsion: 220, edgeLength: 110, gravity: 0.1 },
        animationDurationUpdate: 500,
      },
    ],
  }
})

function onClick(params) {
  if (params.dataType === 'node') {
    emit('node-click', params.data)
  }
}
</script>

<template>
  <v-chart class="chart" :option="option" autoresize @click="onClick" />
</template>

<style scoped>
.chart {
  height: 380px;
}
</style>
