<script setup>
import { ref, onMounted, h } from 'vue'
import { useRouter } from 'vue-router'
import { NDataTable, NTag } from 'naive-ui'
import { api } from '@/api/client'
import { useAppStore } from '@/stores/app'
import PageHeader from '@/components/layout/PageHeader.vue'
import ChartCard from '@/components/charts/ChartCard.vue'

const router = useRouter()
const store = useAppStore()

const rows = ref([])
const loading = ref(false)

const fmt4 = (v) => (v == null ? '—' : Number(v).toFixed(4))
const fmt3 = (v) => (v == null ? '—' : Number(v).toFixed(3))

const columns = [
  { title: 'run', key: 'run_id', width: 200, fixed: 'left' },
  { title: '时序', key: 'temporal_encoder', width: 96 },
  { title: '标签方案', key: 'label_scheme', width: 92, render: (r) => h(NTag, { size: 'small', bordered: false }, { default: () => r.label_scheme || '—' }) },
  { title: 'epochs', key: 'epochs', width: 76 },
  { title: 'R²(prob)', key: 'r2', width: 96, render: (r) => fmt4(r.r2) },
  { title: 'R²(logit)', key: 'r2_logit', width: 96, render: (r) => fmt4(r.r2_logit) },
  { title: 'Spearman', key: 'spearman', width: 96, render: (r) => fmt3(r.spearman) },
  { title: 'IC', key: 'ic', width: 88, render: (r) => fmt3(r.ic) },
  { title: 'AUC', key: 'auc', width: 88, render: (r) => fmt4(r.auc) },
  { title: 'KS', key: 'ks', width: 84, render: (r) => fmt3(r.ks) },
  { title: '评级准确率', key: 'rating_accuracy', width: 100, render: (r) => fmt3(r.rating_accuracy) },
]

function onRowClick(row) {
  store.setSelectedRun(row.run_id)
  router.push('/risk')
}

onMounted(async () => {
  loading.value = true
  try {
    const res = await api.modelExperiments()
    rows.value = res.runs
  } finally {
    loading.value = false
  }
})
</script>

<template>
  <div>
    <PageHeader title="方案对比" subtitle="各训练 run 的配置与指标横向对比，点击行跳转到风险评测" />

    <ChartCard :loading="loading" :empty="!rows.length" empty-text="暂无训练产物">
      <n-data-table
        :columns="columns"
        :data="rows"
        :bordered="false"
        size="small"
        :scroll-x="1200"
        :max-height="620"
        :pagination="{ pageSize: 15 }"
        @row-click="onRowClick"
      />
    </ChartCard>
  </div>
</template>
