<script setup>
defineProps({
  title: { type: String, required: true },
  subtitle: { type: String, default: '' },
})
</script>

<template>
  <div class="page-header">
    <h1 class="title">{{ title }}</h1>
    <p v-if="subtitle" class="subtitle">{{ subtitle }}</p>
    <div class="extra"><slot name="extra" /></div>
  </div>
</template>

<style scoped>
.page-header {
  margin-bottom: 20px;
}

.title {
  margin: 0;
  font-size: clamp(22px, 3vw, 30px);
  font-weight: 700;
  letter-spacing: 0.5px;
}

.subtitle {
  margin: 6px 0 0;
  opacity: 0.65;
  font-size: 14px;
}
</style>
