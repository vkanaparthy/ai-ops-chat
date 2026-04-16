import { defineConfig } from "vite";

const backend = "http://127.0.0.1:8000";

const proxy = {
  "^/ai": { target: backend, changeOrigin: true },
  "^/health": { target: backend, changeOrigin: true },
};

export default defineConfig({
  server: { proxy },
  preview: { proxy },
});
