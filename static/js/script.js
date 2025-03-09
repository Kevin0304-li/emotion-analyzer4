document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const analysisForm = document.getElementById('analysis-form');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const messageContainer = document.getElementById('message-container');
    const newChatBtn = document.getElementById('new-chat');
    const themeToggle = document.getElementById('theme-toggle');
    const emotionPanel = document.getElementById('emotion-panel');
    const closePanel = document.getElementById('close-panel');
    
    // Chart.js instance
    let emotionChart = null;
    
    // Current theme
    let isDarkTheme = false;
    
    // Auto-resize textarea
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        
        // Enable/disable send button based on input
        if (this.value.trim() === '') {
            sendBtn.disabled = true;
        } else {
            sendBtn.disabled = false;
        }
    });
    
    // Theme toggle
    themeToggle.addEventListener('click', function() {
        toggleTheme();
    });
    
    // Toggle theme function
    function toggleTheme() {
        document.body.setAttribute('data-theme', isDarkTheme ? 'light' : 'dark');
        themeToggle.innerHTML = isDarkTheme ? '<i class="fas fa-moon"></i>' : '<i class="fas fa-sun"></i>';
        isDarkTheme = !isDarkTheme;
        
        // Update chart if it exists
        if (emotionChart) {
            updateChartTheme();
        }
    }
    
    // Update chart theme
    function updateChartTheme() {
        const textColor = isDarkTheme ? '#f1f1f1' : '#1a1a1a';
        const gridColor = isDarkTheme ? '#3e3e3e' : '#e5e5e5';
        
        emotionChart.options.scales.x.ticks.color = textColor;
        emotionChart.options.scales.y.ticks.color = textColor;
        emotionChart.options.scales.x.grid.color = gridColor;
        emotionChart.options.scales.y.grid.color = gridColor;
        emotionChart.update();
    }
    
    // Close emotion panel
    closePanel.addEventListener('click', function() {
        emotionPanel.classList.remove('open');
    });
    
    // Handle form submission
    analysisForm.addEventListener('submit', function(e) {
        e.preventDefault();
        const text = userInput.value.trim();
        
        if (text === '') return;
        
        // Add user message to chat
        addUserMessage(text);
        
        // Clear input
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;
        
        // Show loading indicator
        const loadingMessage = addLoadingMessage();
        
        // Send to backend for analysis
        analyzeSentiment(text)
            .then(result => {
                // Remove loading message
                messageContainer.removeChild(loadingMessage);
                
                // Add analysis result
                addAnalysisResult(result);
                
                // Update emotion panel
                updateEmotionPanel(result);
                
                // Open emotion panel
                emotionPanel.classList.add('open');
            })
            .catch(error => {
                // Remove loading message
                messageContainer.removeChild(loadingMessage);
                
                // Add error message
                addErrorMessage(error);
            });
    });
    
    // New chat button
    newChatBtn.addEventListener('click', function() {
        // Clear chat
        while (messageContainer.firstChild) {
            messageContainer.removeChild(messageContainer.firstChild);
        }
        
        // Add welcome message
        addWelcomeMessage();
        
        // Close emotion panel
        emotionPanel.classList.remove('open');
    });
    
    // Add user message to chat
    function addUserMessage(text) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message user';
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>${escapeHtml(text)}</p>
            </div>
        `;
        messageContainer.appendChild(messageDiv);
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
    
    // Add loading message
    function addLoadingMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.innerHTML = `
            <div class="message-content">
                <p><span class="loading-spinner"></span> Analyzing sentiment...</p>
            </div>
        `;
        messageContainer.appendChild(messageDiv);
        messageContainer.scrollTop = messageContainer.scrollHeight;
        return messageDiv;
    }
    
    // Add analysis result to chat
    function addAnalysisResult(result) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message analysis-result';
        
        // Get emotion class
        const emotionClass = result.predicted_emotion.toLowerCase();
        
        // Format confidence as percentage
        const confidence = (result.confidence * 100).toFixed(1);
        
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>Analysis complete:</p>
                <div class="emotion-result">
                    <span class="emotion-label">Primary Emotion:</span>
                    <span class="emotion ${emotionClass}">${capitalize(result.predicted_emotion)}</span>
                    <span class="emotion-confidence">${confidence}% Confidence</span>
                </div>
                <p class="detail-prompt">View detailed analysis in the emotion panel →</p>
            </div>
        `;
        messageContainer.appendChild(messageDiv);
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
    
    // Add error message
    function addErrorMessage(error) {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.innerHTML = `
            <div class="message-content">
                <p class="error-message">Error: ${error.message || 'Failed to analyze sentiment. Please try again.'}</p>
            </div>
        `;
        messageContainer.appendChild(messageDiv);
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
    
    // Add welcome message
    function addWelcomeMessage() {
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message system';
        messageDiv.innerHTML = `
            <div class="message-content">
                <p>Welcome to SentiMind, a context-aware sentiment analysis model that understands the emotional nuances of text beyond just the words.</p>
                <p>Try entering a sentence like "<strong>I will kill you!</strong>" versus "<strong>I will kill you, haha, just kidding my friend.</strong>" to see how context changes the emotional interpretation.</p>
            </div>
        `;
        messageContainer.appendChild(messageDiv);
    }
    
    // Update emotion panel with analysis results
    function updateEmotionPanel(result) {
        document.getElementById('primary-emotion').textContent = capitalize(result.predicted_emotion);
        document.getElementById('confidence-score').textContent = (result.confidence * 100).toFixed(1) + '%';
        
        // Create emotion scores
        const emotionScores = document.getElementById('emotion-scores');
        emotionScores.innerHTML = '';
        
        // Sort emotions by score
        const sortedEmotions = Object.entries(result.emotion_scores).sort((a, b) => b[1] - a[1]);
        
        // Add each emotion score
        sortedEmotions.forEach(([emotion, score]) => {
            const emotionScoreItem = document.createElement('div');
            emotionScoreItem.className = 'emotion-score-item';
            
            // Format score as percentage
            const scorePercentage = (score * 100).toFixed(1);
            
            emotionScoreItem.innerHTML = `
                <span class="emotion-name">${capitalize(emotion)}</span>
                <div class="score-bar-container">
                    <div class="score-bar ${emotion.toLowerCase()}" style="width: ${scorePercentage}%"></div>
                </div>
                <span class="score-value">${scorePercentage}%</span>
            `;
            emotionScores.appendChild(emotionScoreItem);
        });
        
        // Update chart
        updateEmotionChart(sortedEmotions);
    }
    
    // Update emotion chart
    function updateEmotionChart(emotionData) {
        const ctx = document.getElementById('emotion-chart').getContext('2d');
        
        // Prepare data for chart
        const labels = emotionData.map(([emotion]) => capitalize(emotion));
        const data = emotionData.map(([, score]) => score * 100);
        const backgroundColors = emotionData.map(([emotion]) => {
            const colorMap = {
                'joy': '#fdcb6e',
                'sadness': '#74b9ff',
                'anger': '#ff7675',
                'fear': '#a29bfe',
                'surprise': '#ffeaa7',
                'disgust': '#55efc4',
                'trust': '#81ecec',
                'anticipation': '#fab1a0'
            };
            return colorMap[emotion.toLowerCase()] || '#95a5a6';
        });
        
        // Destroy existing chart if it exists
        if (emotionChart) {
            emotionChart.destroy();
        }
        
        // Text color based on theme
        const textColor = isDarkTheme ? '#f1f1f1' : '#1a1a1a';
        const gridColor = isDarkTheme ? '#3e3e3e' : '#e5e5e5';
        
        // Create new chart
        emotionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Emotion Score (%)',
                    data: data,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors,
                    borderWidth: 1
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Score: ${context.raw.toFixed(1)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        ticks: {
                            color: textColor
                        },
                        grid: {
                            color: gridColor
                        }
                    },
                    y: {
                        ticks: {
                            color: textColor
                        },
                        grid: {
                            color: gridColor
                        }
                    }
                }
            }
        });
    }
    
    // Analyze sentiment function
    async function analyzeSentiment(text) {
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ text })
            });
            
            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error('Error analyzing sentiment:', error);
            throw error;
        }
    }
    
    // Helper function to capitalize first letter
    function capitalize(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }
    
    // Helper function to escape HTML
    function escapeHtml(unsafe) {
        return unsafe
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
    
    // Add user avatar placeholder
    const userAvatarPlaceholder = document.createElement('img');
    userAvatarPlaceholder.src = '/static/img/user-avatar.png';
    userAvatarPlaceholder.alt = 'User Avatar';
    userAvatarPlaceholder.className = 'user-avatar';
    userAvatarPlaceholder.style.display = 'none';
    document.body.appendChild(userAvatarPlaceholder);
}); 