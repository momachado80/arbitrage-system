// Configuration
const API_BASE_URL = window.location.hostname === 'localhost' 
    ? 'http://localhost:8000' 
    : window.location.origin;

// State
let currentOpportunities = [];
let isLoading = false;

// DOM Elements
const analyzeBtn = document.getElementById('analyzeBtn');
const refreshBtn = document.getElementById('refreshBtn');
const marketLimitSelect = document.getElementById('marketLimit');
const loadingContainer = document.getElementById('loadingContainer');
const opportunitiesSection = document.getElementById('opportunitiesSection');
const opportunitiesGrid = document.getElementById('opportunitiesGrid');
const emptyState = document.getElementById('emptyState');
const opportunityModal = document.getElementById('opportunityModal');
const modalClose = document.getElementById('modalClose');
const statusIndicator = document.getElementById('statusIndicator');

// Stats elements
const totalMarketsEl = document.getElementById('totalMarkets');
const opportunitiesFoundEl = document.getElementById('opportunitiesFound');
const avgEdgeEl = document.getElementById('avgEdge');
const lastUpdateEl = document.getElementById('lastUpdate');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    checkAPIHealth();
});

// Event Listeners
function setupEventListeners() {
    analyzeBtn.addEventListener('click', handleAnalyze);
    refreshBtn.addEventListener('click', handleAnalyze);
    modalClose.addEventListener('click', closeModal);
    opportunityModal.addEventListener('click', (e) => {
        if (e.target === opportunityModal) {
            closeModal();
        }
    });
    
    // Close modal on ESC key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && opportunityModal.classList.contains('active')) {
            closeModal();
        }
    });
}

// API Health Check
async function checkAPIHealth() {
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 3000);
        
        const response = await fetch(`${API_BASE_URL}/health`, {
            method: 'GET',
            headers: { 'Accept': 'application/json' },
            signal: controller.signal
        });
        
        clearTimeout(timeoutId);
        
        if (response.ok) {
            await response.json(); // Verificar se é JSON válido
            updateStatusIndicator(true);
            return true;
        } else {
            updateStatusIndicator(false);
            return false;
        }
    } catch (error) {
        updateStatusIndicator(false);
        if (error.name !== 'AbortError') {
            console.error('API health check failed:', error);
        }
        return false;
    }
}

function updateStatusIndicator(connected) {
    const dot = statusIndicator.querySelector('.status-dot');
    const text = statusIndicator.querySelector('.status-text');
    
    if (connected) {
        dot.style.background = '#10b981';
        text.textContent = 'Conectado';
    } else {
        dot.style.background = '#ef4444';
        text.textContent = 'Desconectado';
    }
}

// Main Analyze Function - Agora com polling
async function handleAnalyze() {
    if (isLoading) return;
    
    const limit = parseInt(marketLimitSelect.value);
    
    setLoading(true);
    hideEmptyState();
    clearOpportunities();
    
    // Primeiro, testar se a API está acessível (usar /health que sempre existe)
    const isHealthy = await checkAPIHealth();
    if (!isHealthy) {
        let errorMsg = '❌ Não foi possível conectar à API.\n\n';
        errorMsg += '📋 Para resolver:\n\n';
        errorMsg += '1. Abra um terminal e execute:\n';
        errorMsg += '   cd /Users/machado/Desktop/arbitrage-system\n';
        errorMsg += '   uvicorn api:app --host 0.0.0.0 --port 8000 --reload\n\n';
        errorMsg += '2. Aguarde ver "Uvicorn running on http://0.0.0.0:8000"\n\n';
        errorMsg += '3. Teste no navegador: ' + API_BASE_URL + '/health\n\n';
        errorMsg += '4. Depois recarregue esta página (F5)';
        showError(errorMsg);
        setLoading(false);
        return;
    }
    
    try {
        // Iniciar análise (retorna imediatamente com job_id)
        // Usar timeout de 30 segundos para a requisição inicial (deve ser instantânea, mas dar margem)
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 30000); // 30 segundos
        
        let response;
        try {
            response = await fetch(`${API_BASE_URL}/polymarket/analyze?limit=${limit}`, {
                signal: controller.signal,
                method: 'GET',
                headers: {
                    'Accept': 'application/json',
                },
                // Não usar cache
                cache: 'no-cache'
            });
            clearTimeout(timeoutId);
        } catch (fetchError) {
            clearTimeout(timeoutId);
            if (fetchError.name === 'AbortError') {
                throw new Error('Timeout na requisição inicial. Verifique se a API está rodando e acessível.');
            }
            if (fetchError.message.includes('Failed to fetch') || fetchError.message.includes('NetworkError')) {
                throw new Error('Não foi possível conectar à API. Verifique se o servidor está rodando em ' + API_BASE_URL);
            }
            throw fetchError;
        }
        
        if (!response.ok) {
            const errorText = await response.text().catch(() => 'Erro desconhecido');
            throw new Error(`HTTP error! status: ${response.status} - ${errorText}`);
        }
        
        const initialData = await response.json();
        
        // Se já está completo (improvável, mas possível)
        if (initialData.status === 'completed') {
            currentOpportunities = initialData.opportunities || [];
            displayOpportunities(currentOpportunities);
            updateStats(initialData);
            updateStatusIndicator(true);
            setLoading(false);
            return;
        }
        
        // Se está rodando, fazer polling
        if (initialData.job_id) {
            pollAnalysisStatus(initialData.job_id, limit);
            // Não fazer await - deixar polling rodar em background
        } else {
            // Fallback: tentar versão síncrona se não tiver job_id
            console.warn('Job ID não retornado, tentando versão síncrona...');
            // Não fazer nada, deixar o loading continuar e mostrar erro depois
            setTimeout(() => {
                if (isLoading) {
                    showError('Timeout: A análise está demorando muito. Tente novamente com menos mercados.');
                    setLoading(false);
                }
            }, 180000); // 3 minutos
        }
        
    } catch (error) {
        console.error('Erro ao analisar:', error);
        
        // Mensagem de erro mais específica
        let errorMessage = 'Erro ao conectar com a API. ';
        if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
            errorMessage += 'Verifique se o servidor está rodando em ' + API_BASE_URL;
        } else if (error.message.includes('timeout') || error.message.includes('Timeout')) {
            errorMessage += 'A requisição demorou muito. Tente novamente.';
        } else {
            errorMessage += error.message;
        }
        
        showError(errorMessage);
        updateStatusIndicator(false);
        setLoading(false);
    }
}

// Polling do status da análise - Versão melhorada
function pollAnalysisStatus(jobId, limit) {
    const maxAttempts = 180; // 3 minutos máximo
    let attempts = 0;
    let pollInterval = null;
    let isStopped = false;
    
    const stopPolling = () => {
        isStopped = true;
        if (pollInterval) {
            clearInterval(pollInterval);
            pollInterval = null;
        }
    };
    
    const poll = async () => {
        if (isStopped || !isLoading) {
            stopPolling();
            return;
        }
        
        attempts++;
        
        try {
            // Timeout de 5 segundos por requisição
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch(`${API_BASE_URL}/polymarket/analyze/status/${jobId}`, {
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json',
                }
            });
            
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                if (response.status === 404 && attempts < 10) {
                    // Job não encontrado ainda - pode estar sendo criado
                    return;
                }
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const status = await response.json();
            
            // Atualizar mensagem de loading
            const loadingSubtitle = document.querySelector('.loading-subtitle');
            if (loadingSubtitle) {
                loadingSubtitle.textContent = status.message || `Analisando... (${attempts}s)`;
            }
            
            if (status.status === 'completed') {
                stopPolling();
                if (status.result) {
                    currentOpportunities = status.result.opportunities || [];
                    displayOpportunities(currentOpportunities);
                    updateStats(status.result);
                    updateStatusIndicator(true);
                } else {
                    showError('Análise concluída mas sem resultados');
                }
                setLoading(false);
                return;
            }
            
            if (status.status === 'error') {
                stopPolling();
                showError(status.message || 'Erro na análise');
                setLoading(false);
                return;
            }
            
            // Se ainda está rodando
            if (status.status === 'running') {
                if (attempts >= maxAttempts) {
                    stopPolling();
                    showError('Timeout: Análise demorou muito tempo. Tente com menos mercados.');
                    setLoading(false);
                    return;
                }
                // Continuar polling
            }
            
        } catch (error) {
            console.error('Erro no polling:', error);
            
            if (error.name === 'AbortError') {
                // Timeout na requisição - continuar tentando
                if (attempts < maxAttempts) {
                    return;
                }
            }
            
            if (attempts >= maxAttempts) {
                stopPolling();
                showError('Erro ao verificar status. Verifique se a API está rodando.');
                setLoading(false);
                updateStatusIndicator(false);
            }
        }
    };
    
    // Iniciar polling imediatamente
    poll();
    
    // E depois a cada 1 segundo
    pollInterval = setInterval(poll, 1000);
    
    // Timeout de segurança absoluto
    setTimeout(() => {
        if (!isStopped) {
            stopPolling();
            if (isLoading) {
                showError('Timeout: A análise está demorando muito. Tente novamente.');
                setLoading(false);
            }
        }
    }, (maxAttempts + 10) * 1000);
}

// Display Opportunities
function displayOpportunities(opportunities) {
    if (opportunities.length === 0) {
        showEmptyState();
        return;
    }
    
    opportunitiesSection.style.display = 'block';
    emptyState.classList.remove('active');
    
    opportunitiesGrid.innerHTML = opportunities.map((opp, index) => 
        createOpportunityCard(opp, index + 1)
    ).join('');
    
    // Add click listeners to cards
    document.querySelectorAll('.opportunity-card').forEach((card, index) => {
        card.addEventListener('click', () => {
            showModal(opportunities[index]);
        });
    });
}

function createOpportunityCard(opp, rank) {
    const sideClass = opp.recommended_side.toLowerCase();
    const sideEmoji = opp.recommended_side === 'YES' ? '✅' : '❌';
    
    return `
        <div class="opportunity-card">
            <div class="opportunity-header">
                <div class="opportunity-rank">#${rank}</div>
                <div class="opportunity-title">${escapeHtml(opp.market_title)}</div>
            </div>
            
            <div class="opportunity-metrics">
                <div class="metric">
                    <div class="metric-label">Preço Atual</div>
                    <div class="metric-value">$${opp.polymarket_price.toFixed(4)}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Edge</div>
                    <div class="metric-value positive">${opp.edge_percentage.toFixed(2)}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Probabilidade Esperada</div>
                    <div class="metric-value">${(opp.expected_probability * 100).toFixed(1)}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">EV por $1</div>
                    <div class="metric-value positive">$${opp.expected_value.toFixed(4)}</div>
                </div>
            </div>
            
            <div class="opportunity-side ${sideClass}">
                ${sideEmoji} Apostar em: ${opp.recommended_side}
            </div>
            
            <div class="returns-table">
                <table>
                    <thead>
                        <tr>
                            <th>Investimento</th>
                            <th>Retorno</th>
                            <th>ROI</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>$10</td>
                            <td>$${opp.returns_10.toFixed(2)}</td>
                            <td class="roi-${opp.returns_10 >= 0 ? 'positive' : 'negative'}">${((opp.returns_10 / 10) * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>$20</td>
                            <td>$${opp.returns_20.toFixed(2)}</td>
                            <td class="roi-${opp.returns_20 >= 0 ? 'positive' : 'negative'}">${((opp.returns_20 / 20) * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>$100</td>
                            <td>$${opp.returns_100.toFixed(2)}</td>
                            <td class="roi-${opp.returns_100 >= 0 ? 'positive' : 'negative'}">${((opp.returns_100 / 100) * 100).toFixed(2)}%</td>
                        </tr>
                        <tr>
                            <td>$1,000</td>
                            <td>$${opp.returns_1000.toFixed(2)}</td>
                            <td class="roi-${opp.returns_1000 >= 0 ? 'positive' : 'negative'}">${((opp.returns_1000 / 1000) * 100).toFixed(2)}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
    `;
}

// Modal
function showModal(opportunity) {
    const modalTitle = document.getElementById('modalTitle');
    const modalBody = document.getElementById('modalBody');
    
    modalTitle.textContent = opportunity.market_title;
    
    const timeToResolution = opportunity.time_to_resolution_hours 
        ? `${opportunity.time_to_resolution_hours.toFixed(1)} horas`
        : 'N/A';
    
    modalBody.innerHTML = `
        <div class="modal-section">
            <h4>📊 Análise de Probabilidade</h4>
            <div class="modal-grid">
                <div class="modal-metric">
                    <div class="modal-metric-label">Preço Polymarket</div>
                    <div class="modal-metric-value">$${opportunity.polymarket_price.toFixed(4)}</div>
                    <div style="color: #94a3b8; font-size: 0.875rem; margin-top: 0.5rem;">
                        ${(opportunity.implied_probability * 100).toFixed(2)}% implícito
                    </div>
                </div>
                <div class="modal-metric">
                    <div class="modal-metric-label">Probabilidade Esperada</div>
                    <div class="modal-metric-value" style="color: #10b981;">
                        ${(opportunity.expected_probability * 100).toFixed(2)}%
                    </div>
                </div>
                <div class="modal-metric">
                    <div class="modal-metric-label">Divergência</div>
                    <div class="modal-metric-value" style="color: #f59e0b;">
                        ${(opportunity.divergence * 100).toFixed(2)}%
                    </div>
                </div>
                <div class="modal-metric">
                    <div class="modal-metric-label">Edge</div>
                    <div class="modal-metric-value" style="color: #10b981;">
                        ${opportunity.edge_percentage.toFixed(2)}%
                    </div>
                </div>
            </div>
        </div>
        
        <div class="modal-section">
            <h4>💰 Dados de Mercado</h4>
            <div class="modal-grid">
                <div class="modal-metric">
                    <div class="modal-metric-label">Volume 24h</div>
                    <div class="modal-metric-value">$${formatNumber(opportunity.volume_24h)}</div>
                </div>
                <div class="modal-metric">
                    <div class="modal-metric-label">Liquidez Disponível</div>
                    <div class="modal-metric-value">$${formatNumber(opportunity.liquidity_depth)}</div>
                </div>
                <div class="modal-metric">
                    <div class="modal-metric-label">Tempo até Resolução</div>
                    <div class="modal-metric-value">${timeToResolution}</div>
                </div>
                <div class="modal-metric">
                    <div class="modal-metric-label">Confiança</div>
                    <div class="modal-metric-value">${(opportunity.confidence_score * 100).toFixed(1)}%</div>
                </div>
            </div>
        </div>
        
        <div class="modal-section">
            <h4>🎯 Estratégia Recomendada</h4>
            <div style="background: var(--bg-tertiary); padding: 1.5rem; border-radius: 12px;">
                <div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
                    <div class="opportunity-side ${opportunity.recommended_side.toLowerCase()}" style="margin: 0;">
                        ${opportunity.recommended_side === 'YES' ? '✅' : '❌'} ${opportunity.recommended_side}
                    </div>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Investimento sugerido: <strong style="color: var(--text-primary);">$${opportunity.recommended_investment.toFixed(2)}</strong>
                </p>
                <p style="color: var(--text-secondary);">
                    Expected Value: <strong style="color: #10b981;">$${opportunity.expected_value.toFixed(4)}</strong> por $1 investido
                </p>
            </div>
        </div>
        
        <div class="modal-section">
            <h4>📈 Retornos Projetados</h4>
            <div class="returns-table">
                <table>
                    <thead>
                        <tr>
                            <th>Investimento</th>
                            <th>Retorno Esperado</th>
                            <th>ROI</th>
                            <th>Confiança</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>$10</td>
                            <td>$${opportunity.returns_10.toFixed(2)}</td>
                            <td class="roi-${opportunity.returns_10 >= 0 ? 'positive' : 'negative'}">${((opportunity.returns_10 / 10) * 100).toFixed(2)}%</td>
                            <td>${(opportunity.confidence_score * 100).toFixed(1)}%</td>
                        </tr>
                        <tr>
                            <td>$20</td>
                            <td>$${opportunity.returns_20.toFixed(2)}</td>
                            <td class="roi-${opportunity.returns_20 >= 0 ? 'positive' : 'negative'}">${((opportunity.returns_20 / 20) * 100).toFixed(2)}%</td>
                            <td>${(opportunity.confidence_score * 100).toFixed(1)}%</td>
                        </tr>
                        <tr>
                            <td>$100</td>
                            <td>$${opportunity.returns_100.toFixed(2)}</td>
                            <td class="roi-${opportunity.returns_100 >= 0 ? 'positive' : 'negative'}">${((opportunity.returns_100 / 100) * 100).toFixed(2)}%</td>
                            <td>${(opportunity.confidence_score * 100).toFixed(1)}%</td>
                        </tr>
                        <tr>
                            <td>$1,000</td>
                            <td>$${opportunity.returns_1000.toFixed(2)}</td>
                            <td class="roi-${opportunity.returns_1000 >= 0 ? 'positive' : 'negative'}">${((opportunity.returns_1000 / 1000) * 100).toFixed(2)}%</td>
                            <td>${(opportunity.confidence_score * 100).toFixed(1)}%</td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="modal-section">
            <h4>🔗 Informações Técnicas</h4>
            <div style="background: var(--bg-tertiary); padding: 1rem; border-radius: 8px; font-family: monospace; font-size: 0.875rem; color: var(--text-secondary);">
                Condition ID: ${opportunity.condition_id}<br>
                Detectado em: ${new Date(opportunity.detected_at).toLocaleString('pt-BR')}
            </div>
        </div>
    `;
    
    opportunityModal.classList.add('active');
}

function closeModal() {
    opportunityModal.classList.remove('active');
}

// Stats Update
function updateStats(data) {
    totalMarketsEl.textContent = formatNumber(data.total_markets_scanned);
    opportunitiesFoundEl.textContent = data.opportunities_found;
    
    if (data.opportunities && data.opportunities.length > 0) {
        const avgEdge = data.opportunities.reduce((sum, opp) => sum + opp.edge_percentage, 0) / data.opportunities.length;
        avgEdgeEl.textContent = `${avgEdge.toFixed(2)}%`;
    } else {
        avgEdgeEl.textContent = '-';
    }
    
    lastUpdateEl.textContent = new Date().toLocaleTimeString('pt-BR');
}

// UI Helpers
function setLoading(loading) {
    isLoading = loading;
    if (loading) {
        loadingContainer.classList.add('active');
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<div class="spinner" style="width: 20px; height: 20px; border-width: 2px; margin: 0;"></div> Analisando...';
    } else {
        loadingContainer.classList.remove('active');
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = `
            <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                <path d="M10 2L12.09 7.26L18 8.27L14 12.14L14.91 18.02L10 15.77L5.09 18.02L6 12.14L2 8.27L7.91 7.26L10 2Z" fill="currentColor"/>
            </svg>
            Analisar Mercados
        `;
    }
}

function clearOpportunities() {
    opportunitiesGrid.innerHTML = '';
    opportunitiesSection.style.display = 'none';
}

function showEmptyState() {
    emptyState.classList.add('active');
    opportunitiesSection.style.display = 'none';
}

function hideEmptyState() {
    emptyState.classList.remove('active');
}

function showError(message) {
    alert(message); // You can replace this with a better notification system
}

// Utility Functions
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatNumber(num) {
    return new Intl.NumberFormat('pt-BR', {
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(num);
}

