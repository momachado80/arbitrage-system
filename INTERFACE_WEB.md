# Interface Web - Analisador de Arbitragem Polymarket

## 🎨 Interface Moderna e Responsiva

Uma interface web completa e profissional para análise de oportunidades de arbitragem no Polymarket.

## 📁 Arquivos Criados

- `frontend/polymarket-analyzer.html` - Página principal
- `frontend/polymarket-analyzer.css` - Estilos modernos
- `frontend/polymarket-analyzer.js` - Lógica e integração com API

## 🚀 Como Usar

### 1. Iniciar o Servidor API

```bash
# No diretório raiz do projeto
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Abrir a Interface Web

**Opção A: Abrir diretamente no navegador**
```bash
# Navegue até a pasta frontend
cd frontend

# Abra o arquivo no navegador
open polymarket-analyzer.html
# ou
xdg-open polymarket-analyzer.html  # Linux
```

**Opção B: Servidor HTTP simples (recomendado)**
```bash
# Na pasta frontend
python -m http.server 8080

# Acesse: http://localhost:8080/polymarket-analyzer.html
```

**Opção C: Integrar com FastAPI (servir estáticos)**
Adicione ao `api.py`:
```python
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="frontend"), name="static")
```

## ✨ Características da Interface

### Design Moderno
- ✅ Tema escuro profissional
- ✅ Gradientes e animações suaves
- ✅ Design responsivo (mobile-friendly)
- ✅ Tipografia moderna (Inter font)

### Funcionalidades
- ✅ **Dashboard em tempo real** com estatísticas
- ✅ **Cards interativos** para cada oportunidade
- ✅ **Modal detalhado** com informações completas
- ✅ **Tabelas de retornos** para diferentes investimentos
- ✅ **Indicador de status** da conexão com API
- ✅ **Loading states** durante análise
- ✅ **Empty states** quando não há resultados

### Experiência do Usuário
- ✅ Animações suaves e transições
- ✅ Feedback visual imediato
- ✅ Navegação intuitiva
- ✅ Informações organizadas hierarquicamente
- ✅ Cores semânticas (verde para positivo, vermelho para negativo)

## 📊 Componentes Principais

### 1. Header
- Logo e título
- Botão de atualização
- Indicador de status da conexão

### 2. Hero Section
- Título e descrição
- Cards de estatísticas:
  - Mercados escaneados
  - Oportunidades encontradas
  - Edge médio
  - Última atualização

### 3. Filtros
- Seletor de limite de mercados (50, 100, 200, 500)
- Botão de análise

### 4. Cards de Oportunidades
Cada card mostra:
- Rank (#1, #2, #3)
- Título do mercado
- Preço atual
- Edge percentual
- Probabilidade esperada
- Expected Value
- Lado recomendado (YES/NO)
- Tabela de retornos ($10, $20, $100, $1,000)

### 5. Modal de Detalhes
Ao clicar em um card, abre modal com:
- Análise completa de probabilidade
- Dados de mercado (volume, liquidez, tempo)
- Estratégia recomendada
- Retornos projetados detalhados
- Informações técnicas

## 🎯 Fluxo de Uso

1. **Abrir a página** no navegador
2. **Verificar status** da conexão (indicador no header)
3. **Selecionar limite** de mercados a escanear
4. **Clicar em "Analisar Mercados"**
5. **Aguardar análise** (loading spinner)
6. **Visualizar oportunidades** nos cards
7. **Clicar em um card** para ver detalhes completos no modal
8. **Fechar modal** com X ou ESC

## 🔧 Configuração da API

A interface se conecta automaticamente à API:

- **Local**: `http://localhost:8000`
- **Produção**: URL do servidor atual

Para mudar a URL da API, edite em `polymarket-analyzer.js`:
```javascript
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : window.location.origin;
```

## 📱 Responsividade

A interface é totalmente responsiva:
- **Desktop**: Grid de 3 colunas
- **Tablet**: Grid de 2 colunas
- **Mobile**: 1 coluna, layout otimizado

## 🎨 Personalização

### Cores
Edite as variáveis CSS em `polymarket-analyzer.css`:
```css
:root {
    --primary: #6366f1;
    --secondary: #8b5cf6;
    --success: #10b981;
    --bg-primary: #0f172a;
    /* ... */
}
```

### Fontes
A interface usa a fonte Inter do Google Fonts. Para mudar, edite o link no HTML.

## 🐛 Troubleshooting

### Interface não carrega
- Verifique se os arquivos CSS e JS estão na mesma pasta do HTML
- Abra o console do navegador (F12) para ver erros

### API não conecta
- Certifique-se de que o servidor está rodando em `http://localhost:8000`
- Verifique o indicador de status no header
- Veja erros no console do navegador

### Dados não aparecem
- Verifique se a API retorna dados no formato esperado
- Teste o endpoint diretamente: `http://localhost:8000/polymarket/analyze`

## 📝 Próximas Melhorias

Possíveis melhorias futuras:
- [ ] Gráficos de histórico de preços
- [ ] Filtros avançados (por edge, volume, etc.)
- [ ] Notificações em tempo real
- [ ] Exportação de dados (CSV, PDF)
- [ ] Modo claro/escuro
- [ ] Favoritos/bookmarks
- [ ] Compartilhamento de oportunidades

## 📄 Licença

Interface criada para uso com o sistema de arbitragem Polymarket.




