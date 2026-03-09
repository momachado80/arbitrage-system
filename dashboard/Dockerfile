# Dockerfile determinístico — contorna cache obsoleto do Railway/Nixpacks.
# Railway usa este Dockerfile automaticamente quando presente na raiz do serviço.
FROM node:18-alpine AS base

WORKDIR /app

# Copia apenas manifestos para instalação (permite cache de npm em builds idênticos)
COPY package.json package-lock.json* ./

# Instalação limpa — sem cache de pacotes
RUN npm ci --no-cache --ignore-scripts 2>/dev/null || npm install --no-cache --ignore-scripts

# Copia todo o código fonte (qualquer mudança invalida cache daqui pra baixo)
COPY . .

# Build Next.js
RUN npm run build

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
