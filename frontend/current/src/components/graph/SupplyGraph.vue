<script setup>
import { ref, computed, watch } from 'vue'
import { NSpin } from 'naive-ui'
import VChart from '@/plugins/echarts'
import { api } from '@/api/client'
import { RATING_COLORS } from '@/utils/rating'
import { chartPalette } from '@/utils/chartTheme'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  year: { type: Number, required: true },
})

const emit = defineEmits(['node-click'])

const store = useAppStore()

const graphData = ref({ nodes: [], links: [] })
const loading = ref(false)

function buildNodeDetail(id) {
  const node = graphData.value.nodes.find((n) => n.id === id)
  if (!node) return null
  const edges = graphData.value.links
    .filter((l) => l.source === id || l.target === id)
    .map((l) => ({
      relationship: l.relationship,
      peer: l.source === id ? l.target : l.source,
      proportion: l.proportion ?? null,
    }))
  return {
    id: node.id,
    rating: node.rating || '—',
    prob: node.prob ?? null,
    everSt: node.ever_st ?? false,
    edges,
  }
}

const echartsOption = computed(() => {
  const pal = chartPalette(store.isDark)
  const { nodes, links } = graphData.value
  return {
    tooltip: {
      trigger: 'item',
      backgroundColor: pal.tooltipBg,
      borderColor: pal.tooltipBorder,
      textStyle: { color: pal.tooltipText },
      formatter: (p) => {
        if (p.dataType === 'edge') {
          return `${p.data.source} → ${p.data.target}<br/>${p.data.relationship === 'supply' ? '供应' : '销售'}`
        }
        const d = p.data
        return `${d.id}<br/>评级: ${d.rating || '—'}<br/>违约概率: ${d.prob != null ? (d.prob * 100).toFixed(2) + '%' : '—'}`
      },
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        roam: true,
        draggable: true,
        data: nodes.map((n) => ({
          id: n.id,
          name: n.id,
          value: n.rating || '—',
          prob: n.prob,
          rating: n.rating,
          symbolSize: n.prob != null ? 12 + n.prob * 80 : 16,
          itemStyle: { color: RATING_COLORS[n.rating] || '#8a9099' },
        })),
        links: links.map((l) => ({
          source: l.source,
          target: l.target,
          relationship: l.relationship,
          lineStyle: {
            color: l.relationship === 'supply' ? '#4c8dff' : '#36ad6a',
            opacity: 0.55,
            width: 1.5,
            curveness: 0.06,
          },
        })),
        label: { show: true, position: 'right', fontSize: 10, color: pal.text, formatter: '{b}' },
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: 6,
        emphasis: { focus: 'adjacency', lineStyle: { opacity: 0.9, width: 2 } },
        force: { repulsion: 160, edgeLength: 90, gravity: 0.08 },
        animationDurationUpdate: 500,
      },
    ],
  }
})

function onEchartsClick(params) {
  if (params.dataType === 'node') {
    emit('node-click', buildNodeDetail(params.data.id))
  }
}

async function render() {
  loading.value = true
  try {
    graphData.value = await api.graphByYear(props.year)
  } catch {
    graphData.value = { nodes: [], links: [] }
  } finally {
    loading.value = false
  }
}

watch(() => props.year, render, { immediate: true })
</script>

<template>
  <div class="supply-graph">
    <n-spin :show="loading">
      <div v-if="!loading && !graphData.nodes.length" class="empty-tip">
        该年份无供应链数据
      </div>
      <v-chart
        v-else-if="graphData.nodes.length"
        class="chart"
        :option="echartsOption"
        autoresize
        @click="onEchartsClick"
      />
    </n-spin>
  </div>
</template>

<style scoped>
.supply-graph {
  position: relative;
  width: 100%;
  height: 100%;
}

.chart {
  width: 100%;
  height: calc(100vh - 210px);
  min-height: 420px;
}

.empty-tip {
  height: calc(100vh - 210px);
  display: flex;
  align-items: center;
  justify-content: center;
  opacity: 0.5;
}
</style>
