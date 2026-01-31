/**
 * ZIP Game AI - Interactive Dashboard JavaScript
 */

// API Base URL
const API_BASE = '';

// State
let gameState = {
    isPlaying: false,
    autoPlayInterval: null,
    modelLoaded: false
};

// DOM Elements
const elements = {
    loadModelBtn: document.getElementById('loadModelBtn'),
    modelStatus: document.getElementById('modelStatus'),
    newGameBtn: document.getElementById('newGameBtn'),
    aiMoveBtn: document.getElementById('aiMoveBtn'),
    autoPlayBtn: document.getElementById('autoPlayBtn'),
    gameBoard: document.getElementById('gameBoard'),
    targetBadge: document.getElementById('targetBadge'),
    coverageBadge: document.getElementById('coverageBadge'),
    historyList: document.getElementById('historyList'),
    totalGames: document.getElementById('totalGames'),
    winRate: document.getElementById('winRate'),
    validMoves: document.getElementById('validMoves'),
    illegalMoves: document.getElementById('illegalMoves'),
    optimalRate: document.getElementById('optimalRate'),
    resetStatsBtn: document.getElementById('resetStatsBtn'),
    boardSize: document.getElementById('boardSize'),
    numCount: document.getElementById('numCount'),
    solvableMode: document.getElementById('solvableMode'),
    autoPlaySpeed: document.getElementById('autoPlaySpeed'),
    speedLabel: document.getElementById('speedLabel'),
    resultModal: document.getElementById('resultModal'),
    loadingOverlay: document.getElementById('loadingOverlay'),
    loadingText: document.getElementById('loadingText')
};

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    updateStatus();
});

function setupEventListeners() {
    // Load Model
    elements.loadModelBtn.addEventListener('click', loadModel);
    
    // Game Controls
    elements.newGameBtn.addEventListener('click', newGame);
    elements.aiMoveBtn.addEventListener('click', aiMove);
    elements.autoPlayBtn.addEventListener('click', toggleAutoPlay);
    
    // Arrow buttons
    document.querySelectorAll('.arrow-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            const direction = btn.dataset.direction;
            playerMove(direction);
        });
    });
    
    // Keyboard controls
    document.addEventListener('keydown', handleKeyboard);
    
    // Reset stats
    elements.resetStatsBtn.addEventListener('click', resetStats);
    
    // Speed slider
    elements.autoPlaySpeed.addEventListener('input', (e) => {
        elements.speedLabel.textContent = `${e.target.value}ms`;
    });
}

function handleKeyboard(e) {
    const keyMap = {
        'ArrowUp': 'up',
        'ArrowDown': 'down',
        'ArrowLeft': 'left',
        'ArrowRight': 'right'
    };
    
    if (keyMap[e.key]) {
        e.preventDefault();
        playerMove(keyMap[e.key]);
    }
}

// API Calls
async function apiCall(endpoint, method = 'GET', data = null) {
    const options = {
        method,
        headers: { 'Content-Type': 'application/json' }
    };
    
    if (data) {
        options.body = JSON.stringify(data);
    }
    
    const response = await fetch(`${API_BASE}${endpoint}`, options);
    return response.json();
}

async function loadModel() {
    showLoading('Model yükleniyor... Bu işlem birkaç dakika sürebilir.');
    
    try {
        const result = await apiCall('/api/load_model', 'POST', {});
        
        if (result.success) {
            gameState.modelLoaded = true;
            updateModelStatus(true);
            elements.aiMoveBtn.disabled = false;
            elements.autoPlayBtn.disabled = false;
            showNotification('Model başarıyla yüklendi!', 'success');
        } else {
            showNotification('Model yüklenemedi: ' + result.error, 'error');
        }
    } catch (error) {
        showNotification('Bağlantı hatası: ' + error.message, 'error');
    }
    
    hideLoading();
}

async function newGame() {
    const size = parseInt(elements.boardSize.value);
    const numCount = parseInt(elements.numCount.value);
    const solvable = elements.solvableMode.checked;
    
    try {
        const result = await apiCall('/api/new_game', 'POST', {
            size,
            num_count: numCount,
            solvable
        });
        
        if (result.success) {
            renderBoard(result.observation, size);
            updateGameInfo(result.observation);
            clearHistory();
            enableArrowButtons(result.observation.legal_actions);
            gameState.isPlaying = true;
            
            if (result.solution_length > 0) {
                showNotification(`Yeni oyun başladı! (Çözüm: ${result.solution_length} hamle)`, 'success');
            } else {
                showNotification('Yeni oyun başladı!', 'success');
            }
        }
    } catch (error) {
        showNotification('Oyun başlatılamadı: ' + error.message, 'error');
    }
}

async function aiMove() {
    if (!gameState.modelLoaded) {
        showNotification('Önce modeli yükleyin!', 'warning');
        return;
    }
    
    elements.aiMoveBtn.disabled = true;
    
    try {
        const result = await apiCall('/api/ai_move', 'POST', {});
        
        if (result.success) {
            const size = result.observation.board.length;
            renderBoard(result.observation, size);
            updateGameInfo(result.observation);
            updateStats(result.stats);
            addHistoryItem(result.move);
            enableArrowButtons(result.observation.legal_actions);
            
            if (result.move.game_over) {
                handleGameOver(result.move.won);
            }
        } else {
            if (result.game_over) {
                handleGameOver(false);
            } else {
                showNotification('AI hamle yapamadı: ' + result.error, 'error');
            }
        }
    } catch (error) {
        showNotification('Hata: ' + error.message, 'error');
    }
    
    elements.aiMoveBtn.disabled = !gameState.modelLoaded;
}

async function playerMove(direction) {
    try {
        const result = await apiCall('/api/player_move', 'POST', { direction });
        
        if (result.success) {
            const size = result.observation.board.length;
            renderBoard(result.observation, size);
            updateGameInfo(result.observation);
            addHistoryItem(result.move);
            enableArrowButtons(result.observation.legal_actions);
            
            if (result.move.game_over) {
                handleGameOver(result.move.won);
            }
        } else {
            showNotification(result.error, 'warning');
        }
    } catch (error) {
        showNotification('Hata: ' + error.message, 'error');
    }
}

function toggleAutoPlay() {
    if (gameState.autoPlayInterval) {
        // Stop auto play
        clearInterval(gameState.autoPlayInterval);
        gameState.autoPlayInterval = null;
        elements.autoPlayBtn.innerHTML = '<span class="btn-icon">▶️</span>Otomatik Oyna';
        elements.autoPlayBtn.classList.remove('btn-danger');
        elements.autoPlayBtn.classList.add('btn-secondary');
    } else {
        // Start auto play
        const speed = parseInt(elements.autoPlaySpeed.value);
        
        gameState.autoPlayInterval = setInterval(async () => {
            await aiMove();
            
            // Get current status
            const status = await apiCall('/api/status');
            if (!status.observation || status.observation.legal_actions.length === 0) {
                toggleAutoPlay(); // Stop
            }
        }, speed);
        
        elements.autoPlayBtn.innerHTML = '<span class="btn-icon">⏹️</span>Durdur';
        elements.autoPlayBtn.classList.remove('btn-secondary');
        elements.autoPlayBtn.classList.add('btn-danger');
    }
}

async function resetStats() {
    try {
        await apiCall('/api/reset_stats', 'POST', {});
        updateStats({
            total_games: 0,
            wins: 0,
            valid_moves: 0,
            illegal_moves: 0,
            optimal_moves: 0
        });
        showNotification('İstatistikler sıfırlandı', 'success');
    } catch (error) {
        showNotification('Hata: ' + error.message, 'error');
    }
}

async function updateStatus() {
    try {
        const status = await apiCall('/api/status');
        
        gameState.modelLoaded = status.model_loaded;
        updateModelStatus(status.model_loaded);
        updateStats(status.stats);
        
        if (status.observation) {
            const size = status.observation.board.length;
            renderBoard(status.observation, size);
            updateGameInfo(status.observation);
            enableArrowButtons(status.observation.legal_actions);
        }
        
        if (status.model_loaded) {
            elements.aiMoveBtn.disabled = false;
            elements.autoPlayBtn.disabled = false;
        }
    } catch (error) {
        console.log('Status check failed:', error);
    }
}

// UI Updates
function renderBoard(obs, size) {
    const board = obs.board;
    const currentPos = obs.current_pos;
    const visited = obs.visited.map(v => `${v[0]},${v[1]}`);
    const currentTarget = obs.current_target;
    
    elements.gameBoard.innerHTML = '';
    elements.gameBoard.style.gridTemplateColumns = `repeat(${size}, 55px)`;
    
    for (let r = 0; r < size; r++) {
        for (let c = 0; c < size; c++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            
            const posKey = `${r},${c}`;
            const cellValue = board[r][c];
            
            if (r === currentPos[0] && c === currentPos[1]) {
                cell.classList.add('current');
                cell.textContent = '★';
            } else if (visited.includes(posKey)) {
                cell.classList.add('visited');
            } else if (cellValue > 0) {
                if (cellValue === currentTarget) {
                    cell.classList.add('target');
                } else {
                    cell.classList.add('number');
                }
                cell.textContent = cellValue;
            }
            
            elements.gameBoard.appendChild(cell);
        }
    }
}

function updateGameInfo(obs) {
    elements.targetBadge.textContent = `Hedef: ${obs.current_target}`;
    
    const size = obs.board.length;
    const coverage = (obs.visited.length / (size * size) * 100).toFixed(0);
    elements.coverageBadge.textContent = `Kapsam: ${coverage}%`;
}

function updateModelStatus(loaded) {
    const dot = elements.modelStatus.querySelector('.status-dot');
    const text = elements.modelStatus.querySelector('.status-text');
    
    if (loaded) {
        dot.classList.remove('offline');
        dot.classList.add('online');
        text.textContent = 'Model Hazır';
    } else {
        dot.classList.remove('online');
        dot.classList.add('offline');
        text.textContent = 'Model Yüklenmedi';
    }
}

function updateStats(stats) {
    elements.totalGames.textContent = stats.total_games || 0;
    
    const winRate = stats.total_games > 0 
        ? ((stats.wins / stats.total_games) * 100).toFixed(0) 
        : 0;
    elements.winRate.textContent = `${winRate}%`;
    
    elements.validMoves.textContent = stats.valid_moves || 0;
    elements.illegalMoves.textContent = stats.illegal_moves || 0;
    
    const totalMoves = (stats.valid_moves || 0) + (stats.illegal_moves || 0);
    const optimalRate = totalMoves > 0 
        ? ((stats.optimal_moves / totalMoves) * 100).toFixed(0) 
        : 0;
    elements.optimalRate.textContent = `${optimalRate}%`;
}

function addHistoryItem(move) {
    // Clear empty message
    if (elements.historyList.querySelector('.history-empty')) {
        elements.historyList.innerHTML = '';
    }
    
    const item = document.createElement('div');
    item.className = 'history-item';
    
    let actionClass = 'valid';
    let actionText = move.action.toUpperCase();
    
    if (!move.valid) {
        actionClass = 'invalid';
        if (move.fallback) {
            actionText += ` → ${move.fallback.toUpperCase()}`;
        }
    } else if (move.optimal) {
        actionClass = 'optimal';
        actionText += ' ⭐';
    }
    
    item.innerHTML = `
        <span class="turn">#${move.turn}</span>
        <span class="action ${actionClass}">${actionText}</span>
    `;
    
    elements.historyList.insertBefore(item, elements.historyList.firstChild);
    
    // Keep only last 20
    while (elements.historyList.children.length > 20) {
        elements.historyList.removeChild(elements.historyList.lastChild);
    }
}

function clearHistory() {
    elements.historyList.innerHTML = '<div class="history-empty"><span>Henüz hamle yapılmadı</span></div>';
}

function enableArrowButtons(legalActions) {
    document.querySelectorAll('.arrow-btn').forEach(btn => {
        const direction = btn.dataset.direction;
        btn.disabled = !legalActions.includes(direction);
    });
}

function handleGameOver(won) {
    gameState.isPlaying = false;
    
    // Stop auto play if running
    if (gameState.autoPlayInterval) {
        toggleAutoPlay();
    }
    
    // Show modal
    const modal = elements.resultModal;
    const icon = document.getElementById('resultIcon');
    const title = document.getElementById('resultTitle');
    const message = document.getElementById('resultMessage');
    
    if (won) {
        icon.textContent = '🏆';
        title.textContent = 'Tebrikler!';
        message.textContent = 'Oyunu başarıyla kazandınız!';
    } else {
        icon.textContent = '😔';
        title.textContent = 'Oyun Bitti';
        message.textContent = 'Maalesef bu sefer olmadı.';
    }
    
    modal.classList.add('active');
}

function closeModal() {
    elements.resultModal.classList.remove('active');
}

function showLoading(text) {
    elements.loadingText.textContent = text;
    elements.loadingOverlay.classList.add('active');
}

function hideLoading() {
    elements.loadingOverlay.classList.remove('active');
}

function showNotification(message, type = 'info') {
    // Create toast notification
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    toast.style.cssText = `
        position: fixed;
        bottom: 20px;
        right: 20px;
        padding: 14px 24px;
        background: ${type === 'success' ? '#10b981' : type === 'error' ? '#ef4444' : type === 'warning' ? '#f59e0b' : '#6366f1'};
        color: white;
        border-radius: 8px;
        font-size: 0.9rem;
        font-weight: 500;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        z-index: 3000;
        animation: slideInToast 0.3s ease;
    `;
    
    document.body.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutToast 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// Add toast animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInToast {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOutToast {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
    .btn-danger {
        background: linear-gradient(135deg, #ef4444, #dc2626) !important;
    }
`;
document.head.appendChild(style);

// Expose closeModal globally
window.closeModal = closeModal;
