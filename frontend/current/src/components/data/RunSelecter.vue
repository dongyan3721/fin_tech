<script setup>
import { computed } from 'vue'
import { NSelect } from 'naive-ui'
import { useAppStore } from '@/stores/app'

const props = defineProps({
  runs: { type: Array, default: () => [] },
})

const store = useAppStore()

const options = computed(() =>
  props.runs.map((r) => ({ label: r, value: r })),
)
</script>

<template>
  <n-select
    v-model:value="store.selectedRun"
    :options="options"
    placeholder="选择训练 run"
    filterable
    size="small"
    class="run-selecter"
  />
</template>

<style scoped>
.run-selecter {
  min-width: 240px;
  max-width: 320px;
}
</style>
