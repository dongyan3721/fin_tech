<script setup>
import { ref, computed, watch, onMounted, h } from 'vue'
import { useRoute } from 'vue-router'
import {
  NGrid, NGi, NSelect, NSpin, NEmpty, NCard,
} from 'naive-ui'
import { api } from '@/api/client'
import PageHeader from '@/components/layout/PageHeader.vue'
import MetricCard from '@/components/data/MetricCard.vue'
import RatingTag from '@/components/data/RatingTag.vue'
import ChartCard from '@/components/charts/ChartCard.vue'
import TrendLines from '@/components/charts/TrendLines.vue'
import SubGraph from '@/components/graph/SubGraph.vue'
import { fmtProb } from '@/utils/rating'

const route = useRoute()

const FEATURES = [
  { key: 'debt_to_asset_ratio', label: '资产负债率' },
  { key: 'current_ratio', label: '流动比率' },
  { key: 'quick_ratio', label: '速动比率' },
  { key: 'interest_coverage_ratio', label: '利息保障倍数' },
  { key: 'total_assets', label: '总资产' },
  { key: 'total_liab', label: '总负债' },
  { key: 'current_assets', label: '流动资产' },
  { key: 'current_liab', label: '流动负债' },
  { key: 'revenue', label: '营业收入' },
  { key: 'operate_profit', label: '营业利润' },
]

const selectedSymbol = ref(null)
const searchOptions = ref([])
const searching = ref(false)
const loading = ref(false)
const detail = ref(null)
const selectedFeature = ref('debt_to_asset_ratio')

const featureOptions = FEATURES.map((f) => ({ label: f.label, value: f.key }))

const latestLabel = computed(() => {
  const ls = detail.value?.labels || []
  return ls.length ? ls[ls.length - 1] : null
})

const everSt = computed(() => {
  const ls = detail.value?.labels || []
  return ls.some((l) => (l.st_level > 0) || l.delisted === 1)
})

const yearsRange = computed(() => {
  const fs = detail.value?.financial || []
  if (!fs.length) return '—'
  return `${fs[0].year}–${fs[fs.length - 1].year}`
})

const featureTrend = computed(() => {
  const fs = detail.value?.financial || []
  return {
    categories: fs.map((f) => f.year),
    series: [{
      name: FEATURES.find((f) => f.key === selectedFeature.value)?.label || selectedFeature.value,
      data: fs.map((f) => f[selectedFeature.value] ?? null),
      color: '#4c8dff',
    }],
  }
})

const probHistory = computed(() => {
  const ls = detail.value?.labels || []
  const ps = detail.value?.predictions || []
  const years = [...new Set([...ls.map((l) => l.year), ...ps.map((p) => p.year)])].sort((a, b) => a - b)
  const labelMap = Object.fromEntries(ls.map((l) => [l.year, l.default_probability]))
  const predMap = Object.fromEntries(ps.map((p) => [p.year, p.predicted_probability]))
  return {
    categories: years,
    series: [
      { name: '实际(标签)', data: years.map((y) => labelMap[y] ?? null), color: '#36ad6a' },
      { name: '模型预测', data: years.map((y) => predMap[y] ?? null), color: '#e05a5a' },
    ],
  }
})

const subgraph = computed(() => {
  const d = detail.value
  if (!d) return { center: '', nodes: [], links: [] }
  const peerSet = new Set()
  const links = []
  for (const e of d.edges) {
    peerSet.add(e.peer)
    links.push(e.direction === 'out'
      ? { source: d.symbol, target: e.peer, relationship: e.relationship }
      : { source: e.peer, target: d.symbol, relationship: e.relationship })
  }
  const nodes = [{ id: d.symbol, rating: latestLabel.value?.risk_rating }]
  for (const p of peerSet) nodes.push({ id: p, rating: undefined })
  return { center: d.symbol, nodes, links }
})

async function onSearch(q) {
  if (!q) { searchOptions.value = []; return }
  searching.value = true
  try {
    const res = await api.companySearch(q)
    searchOptions.value = res.items.map((i) => ({ label: i.symbol, value: i.symbol }))
  } finally {
    searching.value = false
  }
}

async function loadDetail() {
  if (!selectedSymbol.value) { detail.value = null; return }
  loading.value = true
  try {
    detail.value = await api.companyDetail(selectedSymbol.value)
  } finally {
    loading.value = false
  }
}

watch(selectedSymbol, loadDetail)

onMounted(() => {
  if (route.query.symbol) {
    selectedSymbol.value = String(route.query.symbol).replace(/\D/g, '').padStart(6, '0')
  }
})
</script>

<template>
  <div>
    <PageHeader title="企业分析" subtitle="单公司财务趋势、违约概率历史与供应链子图">
      <template #extra>
        <n-select
          v-model:value="selectedSymbol"
          filterable
          remote
          clearable
          :options="searchOptions"
          :loading="searching"
          placeholder="输入股票代码搜索公司"
          style="width: 240px"
          @search="onSearch"
        />
      </template>
    </PageHeader>

    <n-spin :show="loading">
      <template v-if="detail">
        <!-- 概览 -->
        <n-grid cols="2 s:2 m:3 l:5" :x-gap="16" :y-gap="16" responsive="screen">
          <n-gi><MetricCard label="股票代码" :value="detail.symbol" accent="#4c8dff" /></n-gi>
          <n-gi>
            <n-card size="small" class="overview-card">
              <div class="label">最新评级</div>
              <div class="rating-wrap"><RatingTag :rating="latestLabel?.risk_rating" size="medium" /></div>
            </n-card>
          </n-gi>
          <n-gi><MetricCard label="违约概率(标签)" :value="latestLabel ? fmtProb(latestLabel.default_probability, 3) : '—'" accent="#e05a5a" /></n-gi>
          <n-gi><MetricCard label="财务区间" :value="yearsRange" accent="#36ad6a" /></n-gi>
          <n-gi><MetricCard label="供应链关系" :value="String(detail.n_edges)" :sub="everSt ? '曾 ST/退市' : ''" accent="#f0a020" /></n-gi>
        </n-grid>

        <!-- 图表 -->
        <n-grid cols="1 m:2" :x-gap="16" :y-gap="16" responsive="screen" style="margin-top: 16px">
          <n-gi>
            <ChartCard title="财务特征趋势" :empty="!detail.financial.length">
              <template #header>
                <div class="chart-head">
                  <span class="title">财务特征趋势</span>
                  <n-select v-model:value="selectedFeature" :options="featureOptions" size="small" style="width: 150px" />
                </div>
              </template>
              <TrendLines v-if="detail.financial.length" :categories="featureTrend.categories" :series="featureTrend.series" />
            </ChartCard>
          </n-gi>
          <n-gi>
            <ChartCard title="违约概率历史（标签 vs 预测）" :empty="!detail.labels.length && !detail.predictions.length">
              <TrendLines v-if="probHistory.categories.length" :categories="probHistory.categories" :series="probHistory.series" y-name="概率" />
            </ChartCard>
          </n-gi>
        </n-grid>

        <!-- 子图 -->
        <ChartCard title="供应链子图（点击邻居可跳转）" :empty="!detail.edges.length" style="margin-top: 16px">
          <SubGraph
            v-if="detail.edges.length"
            :center="subgraph.center"
            :nodes="subgraph.nodes"
            :links="subgraph.links"
            @node-click="(n) => { if (n.id !== detail.symbol) selectedSymbol = n.id }"
          />
        </ChartCard>
      </template>
      <n-empty v-else class="empty-tip" description="请在上方搜索并选择一家公司" />
    </n-spin>
  </div>
</template>

<style scoped>
.chart-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}

.title {
  font-weight: 600;
  font-size: 15px;
}

.overview-card {
  border-radius: 12px;
}

.label {
  font-size: 13px;
  opacity: 0.62;
}

.rating-wrap {
  margin-top: 8px;
}

.empty-tip {
  margin-top: 12vh;
}
</style>
