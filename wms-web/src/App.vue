<template>
  <LoginPage v-if="!loggedIn" @login-success="onLogin" />
  <ChatDialog v-else />
</template>

<script setup>
import { ref } from 'vue'
import LoginPage from './components/LoginPage.vue'
import ChatDialog from './components/ChatDialog.vue'

const loggedIn = ref(!!sessionStorage.getItem('token'))

function onLogin() {
  loggedIn.value = true
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
</style>
