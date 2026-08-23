<script setup>
import { ref, onMounted } from 'vue'
import { NInput, NButton, useMessage } from 'naive-ui'
import { api } from '@/api/client'
import PageHeader from '@/components/layout/PageHeader.vue'
import YearPicker from '@/components/graph/YearPicker.vue'
import GraphLegend from '@/components/graph/GraphLegend.vue'
import SupplyGraph from '@/components/graph/SupplyGraph.vue'
import NodeDrawer from '@/components/graph/NodeDrawer.vue'

const message = useMessage()

const years = ref([])
const minYear = ref(2001)
const maxYear = ref(2025)
const year = ref(null)

const drawerShow = ref(false)
const drawerNode = ref(null)

const searchText = ref('')

onMounted(async () => {
  const res = await api.graphYears()
  years.value = res.years || []
  if (years.value.length) {
    minYear.value = years.value[0].year
    maxYear.value = years.value[years.value.length - 1].year
    year.value = maxYear.value
  }
})

function onNodeClick(node) {
  drawerNode.value = node
  drawerShow.value = true
}

async function onSearch() {
  const q = searchText.value.trim()
  if (!q) return
  try {
    const r = await api.graphLocate(q)
    if (!r.years.length) {
      message.warning(`未在供应链中找到 ${r.symbol}`)
      return
    }
    year.value = r.latest
    message.info(`${r.symbol} 最近出现于 ${r.latest} 年（共 ${r.years.length} 个年份）`)
    const g = await api.graphByYear(r.latest)
    const node = g.nodes.find((n) => n.id === r.symbol)
    if (node) {
      const edges = g.links
        .filter((l) => l.source === r.symbol || l.target === r.symbol)
        .map((l) => ({
          relationship: l.relationship,
          peer: l.source === r.symbol ? l.target : l.source,
          proportion: l.proportion ?? null,
        }))
      onNodeClick({ id: node.id, rating: node.rating || '—', prob: node.prob ?? null, everSt: node.ever_st ?? false, edges })
    }
  } catch (e) {
    message.error(`搜索失败：${e.message}`)
  }
}
</script>

<template>
  <div>
    <PageHeader title="供应链图谱" subtitle="按年份切换的供应链网络" />

    <div class="toolbar">
      <YearPicker v-if="year != null" v-model:year="year" :min="minYear" :max="maxYear" />
      <div class="search">
        <n-input
          v-model:value="searchText"
          size="small"
          placeholder="输入股票代码定位公司"
          style="width: 200px"
          @keyup.enter="onSearch"
        />
        <n-button size="small" type="primary" ghost @click="onSearch">定位</n-button>
      </div>
      <GraphLegend />
    </div>

    <div class="graph-wrap">
      <SupplyGraph
        v-if="year != null"
        :year="year"
        @node-click="onNodeClick"
      />
    </div>

    <NodeDrawer v-model:show="drawerShow" :node="drawerNode" />
  </div>
</template>

<style scoped>
.toolbar {
  display: flex;
  align-items: center;
  gap: 20px;
  flex-wrap: wrap;
  margin-bottom: 12px;
  padding: 10px 14px;
  border-radius: 12px;
  background: rgba(128, 138, 157, 0.08);
}

.search {
  display: flex;
  gap: 8px;
  align-items: center;
}

.graph-wrap {
  height: calc(100vh - 190px);
  min-height: 460px;
  border-radius: 12px;
  border: 1px solid rgba(128, 138, 157, 0.16);
  overflow: hidden;
}
</style>
