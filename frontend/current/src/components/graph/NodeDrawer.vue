<script setup>
import { computed } from 'vue'
import { useRouter } from 'vue-router'
import { NDrawer, NDrawerContent, NDescriptions, NDescriptionsItem, NList, NListItem, NEmpty, NButton } from 'naive-ui'
import RatingTag from '@/components/data/RatingTag.vue'
import { fmtProb } from '@/utils/rating'

const props = defineProps({
  show: { type: Boolean, default: false },
  node: { type: Object, default: null },
})

const emit = defineEmits(['update:show'])
const router = useRouter()

const show = computed({
  get: () => props.show,
  set: (v) => emit('update:show', v),
})

const edges = computed(() => props.node?.edges || [])

function goCompany() {
  if (props.node?.id) {
    show.value = false
    router.push(`/company?symbol=${props.node.id}`)
  }
}
</script>

<template>
  <n-drawer v-model:show="show" :width="360" placement="right">
    <n-drawer-content title="节点详情" closable>
      <template v-if="node">
        <div class="node-head">
          <span class="symbol">{{ node.id }}</span>
          <RatingTag :rating="node.rating" />
        </div>
        <n-descriptions label-placement="left" :column="1" size="small" class="desc">
          <n-descriptions-item label="违约概率">
            {{ fmtProb(node.prob, 3) }}
          </n-descriptions-item>
          <n-descriptions-item label="评级">
            {{ node.rating || '—' }}
          </n-descriptions-item>
          <n-descriptions-item label="风险事件">
            {{ node.everSt ? '曾 ST / 退市' : '无' }}
          </n-descriptions-item>
        </n-descriptions>

        <div class="subtitle">直接上下游（{{ edges.length }}）</div>
        <n-empty v-if="!edges.length" description="暂无上下游" size="small" />
        <n-list v-else hoverable clickable size="small">
          <n-list-item v-for="(e, i) in edges" :key="i">
            <div class="edge-row">
              <span class="edge-type" :class="e.relationship">{{ e.relationship === 'supply' ? '↑ 供应商' : '↓ 客户' }}</span>
              <span class="edge-peer">{{ e.peer }}</span>
              <span v-if="e.proportion != null" class="edge-prop">{{ Number(e.proportion).toFixed(1) }}%</span>
            </div>
          </n-list-item>
        </n-list>

        <n-button type="primary" block ghost style="margin-top: 16px" @click="goCompany">
          查看企业分析
        </n-button>
      </template>
      <n-empty v-else description="点击节点查看详情" />
    </n-drawer-content>
  </n-drawer>
</template>

<style scoped>
.node-head {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 12px;
}

.symbol {
  font-size: 22px;
  font-weight: 700;
  font-variant-numeric: tabular-nums;
}

.desc {
  margin-bottom: 16px;
}

.subtitle {
  font-weight: 600;
  margin-bottom: 8px;
  font-size: 13px;
}

.edge-row {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 13px;
}

.edge-type {
  width: 72px;
  opacity: 0.75;
}

.edge-type.supply {
  color: #4c8dff;
}

.edge-type.sale {
  color: #36ad6a;
}

.edge-peer {
  flex: 1;
  font-variant-numeric: tabular-nums;
}

.edge-prop {
  opacity: 0.6;
}
</style>
