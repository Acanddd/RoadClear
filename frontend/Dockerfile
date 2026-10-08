FROM node:22.14.0-bookworm-slim AS build
WORKDIR /app
RUN npm install -g pnpm@11.19.0
COPY frontend/vue/package.json frontend/vue/pnpm-lock.yaml frontend/vue/pnpm-workspace.yaml ./
RUN pnpm install --frozen-lockfile
COPY frontend/vue/ ./
RUN pnpm build
FROM nginx:1.27.4-alpine
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
