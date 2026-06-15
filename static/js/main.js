document.addEventListener('DOMContentLoaded', () => {
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const previewContainer = document.getElementById('preview-container');
    const previewImg = document.getElementById('preview-img');
    const uploadPrompt = document.getElementById('upload-prompt');
    const classifyBtn = document.getElementById('classify-btn');
    const loaderContainer = document.getElementById('loader-container');
    const resultCard = document.getElementById('result-card');
    const errorAlert = document.getElementById('error-alert');
    
    let selectedFile = null;
    let probabilityChart = null;

    // Trigger file dialog on upload zone click
    uploadZone.addEventListener('click', () => {
        fileInput.click();
    });

    // Handle drag events
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        uploadZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            uploadZone.classList.remove('dragover');
        }, false);
    });

    // Handle dropped files
    uploadZone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    }, false);

    // Handle file selection from input
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    // Process and preview selected file
    function handleFile(file) {
        // Validate image file type
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/webp', 'image/gif'];
        if (!allowedTypes.includes(file.type)) {
            showError("Format file tidak didukung. Silakan unggah file gambar (PNG, JPG, JPEG, WEBP).");
            return;
        }

        selectedFile = file;
        hideError();
        resultCard.style.display = 'none';

        // Read and preview image
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewContainer.style.display = 'block';
            uploadPrompt.style.display = 'none';
            classifyBtn.removeAttribute('disabled');
            
            // Scroll to preview/button smoothly
            classifyBtn.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        };
        reader.readAsDataURL(file);
    }

    // Handle classification request
    classifyBtn.addEventListener('click', () => {
        if (!selectedFile) return;

        // Reset UI states
        hideError();
        resultCard.style.display = 'none';
        loaderContainer.style.display = 'block';
        classifyBtn.setAttribute('disabled', 'true');
        
        // Scroll to loader
        loaderContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        const formData = new FormData();
        formData.append('file', selectedFile);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            loaderContainer.style.display = 'none';
            classifyBtn.removeAttribute('disabled');

            if (data.success) {
                displayResults(data);
            } else {
                showError(data.error || "Terjadi kesalahan saat memproses prediksi.");
            }
        })
        .catch(err => {
            console.error(err);
            loaderContainer.style.display = 'none';
            classifyBtn.removeAttribute('disabled');
            showError("Tidak dapat terhubung ke server. Pastikan aplikasi Flask sedang berjalan.");
        });
    });

    // Translate class names dictionary
    const classTranslation = {
        'ADONIS': 'Kupu-kupu Biru Adonis',
        'MESTRA': 'Kupu-kupu Pale Mestra',
        'MONARCH': 'Kupu-kupu Raja (Monarch)',
        'PEACOCK': 'Kupu-kupu Merak (Peacock)',
        'ZEBRA_LONG_WING': 'Kupu-kupu Sayap Panjang Zebra'
    };

    // Render prediction results to UI
    function displayResults(data) {
        // Show target output elements
        resultCard.style.display = 'flex';
        
        // Populate species metadata
        document.getElementById('res-image').src = data.image_url;
        document.getElementById('res-class-name').textContent = data.details.common_name;
        document.getElementById('res-scientific-name').textContent = data.details.scientific_name;
        document.getElementById('res-confidence').textContent = `Kepercayaan ${(data.confidence * 100).toFixed(1)}%`;
        
        // Populate info pills
        document.getElementById('res-desc').textContent = data.details.description;
        document.getElementById('res-habitat').textContent = data.details.habitat;
        document.getElementById('res-funfact').textContent = data.details.fun_fact;

        // Update confidence progress bar
        const mainBar = document.getElementById('res-main-progress');
        mainBar.style.width = `${(data.confidence * 100)}%`;

        // Handle Fallback Mode message
        const fallbackAlert = document.getElementById('fallback-mode-alert');
        if (data.fallback) {
            fallbackAlert.classList.remove('d-none');
        } else {
            fallbackAlert.classList.add('d-none');
        }

        // Render species distribution chart using Chart.js
        renderChart(data.probabilities);

        // Scroll to results smoothly
        resultCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Chart.js helper to show probabilities
    function renderChart(probabilities) {
        const ctx = document.getElementById('probabilities-chart').getContext('2d');
        
        // Sort classes by probability value and translate names
        const sortedProbabilities = Object.entries(probabilities)
            .map(([name, val]) => {
                const translatedName = classTranslation[name] || name.replace(/_/g, ' ');
                return { name: translatedName, val: val * 100 };
            })
            .sort((a, b) => b.val - a.val);

        const labels = sortedProbabilities.map(item => item.name);
        const dataValues = sortedProbabilities.map(item => item.val);

        // Clear pre-existing charts to avoid overlapping glitches
        if (probabilityChart) {
            probabilityChart.destroy();
        }

        // Create new horizontal bar chart
        probabilityChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Kepercayaan (%)',
                    data: dataValues,
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.85)', // First place gets main primary color
                        'rgba(59, 130, 246, 0.6)',
                        'rgba(148, 163, 184, 0.4)',
                        'rgba(148, 163, 184, 0.4)',
                        'rgba(148, 163, 184, 0.4)'
                    ],
                    borderColor: [
                        '#10b981',
                        '#3b82f6',
                        '#94a3b8',
                        '#94a3b8',
                        '#94a3b8'
                    ],
                    borderWidth: 1.5,
                    borderRadius: 6,
                    barPercentage: 0.65
                }]
            },
            options: {
                indexAxis: 'y', // Makes the bar chart horizontal
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return ` ${context.parsed.x.toFixed(2)}%`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.05)'
                        },
                        ticks: {
                            color: '#94a3b8',
                            font: {
                                family: 'Inter'
                            }
                        }
                    },
                    y: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#f8fafc',
                            font: {
                                family: 'Outfit',
                                weight: 500,
                                size: 13
                            }
                        }
                    }
                }
            }
        });
    }

    // Alert helper functions
    function showError(message) {
        errorAlert.textContent = message;
        errorAlert.classList.remove('d-none');
        errorAlert.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Handle hide error
    function hideError() {
        errorAlert.classList.add('d-none');
    }
});
