<script setup>
import { computed } from 'vue'
import { NTag } from 'naive-ui'
import { ratingColor } from '@/utils/rating'

const props = defineProps({
  rating: { type: String, default: '' },
  size: { type: String, default: 'small' },
})

const color = computed(() => ratingColor(props.rating))
</script>

<template>
  <n-tag
    v-if="rating && rating !== '—'"
    :size="size"
    :bordered="false"
    round
    :style="{ color, background: color + '22', fontWeight: 600 }"
  >
    {{ rating }}
  </n-tag>
  <span v-else class="muted">—</span>
</template>

<style scoped>
.muted {
  opacity: 0.45;
}
</style>
