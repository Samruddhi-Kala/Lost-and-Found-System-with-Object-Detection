// static/app.js
document.addEventListener('DOMContentLoaded', () => {
    // Mobile menu functionality
    initializeMobileMenu();
    // Initialize alert dismissal
    initializeAlerts();
    // Initialize form validation
    initializeForms();
    // Initialize file upload previews
    initializeFileUploads();
    // Initialize smooth scroll
    initializeSmoothScroll();
});

// Mobile menu functionality
function initializeMobileMenu() {
    const menuBtn = document.querySelector('.mobile-menu-btn');
    const navLinks = document.querySelector('.nav-links');
    
    if (menuBtn) {
        menuBtn.addEventListener('click', () => {
            navLinks.classList.toggle('show');
            menuBtn.classList.toggle('active');
        });
    }
}

// Alert management
function initializeAlerts() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        const closeBtn = alert.querySelector('.alert-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                alert.style.animation = 'fadeOut 0.3s ease-out forwards';
                setTimeout(() => alert.remove(), 300);
            });
        }
    });
}

// Form validation
function initializeForms() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        const inputs = form.querySelectorAll('input[required], textarea[required]');
        
        inputs.forEach(input => {
            input.addEventListener('input', () => {
                validateInput(input);
            });
        });

        form.addEventListener('submit', handleFormSubmit);
    });
}

function validateInput(input) {
    if (input.validity.valid) {
        input.classList.remove('invalid');
        input.classList.add('valid');
    } else {
        input.classList.remove('valid');
        input.classList.add('invalid');
    }
}

function handleFormSubmit(e) {
    if (!e.target.checkValidity()) {
        e.preventDefault();
        const inputs = e.target.querySelectorAll('input[required], textarea[required]');
        inputs.forEach(input => {
            if (!input.validity.valid) {
                input.classList.add('invalid');
            }
        });
    }
}

// File upload preview
function initializeFileUploads() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    fileInputs.forEach(input => {
        input.addEventListener('change', handleFileSelect);
    });
}

function handleFileSelect(e) {
    const preview = document.createElement('div');
    preview.className = 'file-preview fade-in';
    
    const file = e.target.files[0];
    if (file && file.type.startsWith('image/')) {
        createImagePreview(file, preview, e.target.parentElement);
    }
}

function createImagePreview(file, preview, container) {
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.innerHTML = `
            <div class="preview-container">
                <img src="${e.target.result}" alt="Upload preview" class="upload-preview">
                <p class="file-name">${file.name}</p>
            </div>
        `;
    };
    reader.readAsDataURL(file);
    
    const oldPreview = container.querySelector('.file-preview');
    if (oldPreview) {
        oldPreview.remove();
    }
    container.appendChild(preview);
}

// Smooth scroll
function initializeSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', handleSmoothScroll);
    });
}

function handleSmoothScroll(e) {
    e.preventDefault();
    const targetId = this.getAttribute('href');
    if (targetId === '#') return;
    
    const target = document.querySelector(targetId);
    if (target) {
        target.scrollIntoView({
            behavior: 'smooth',
            block: 'start'
        });
    }
}

// Admin functionality
async function confirmPair() {
    const found = document.getElementById('found_id').value;
    const lost = document.getElementById('lost_id').value;
    
    if (!found || !lost) {
        showNotification('Please enter both IDs', 'error');
        return;
    }

    try {
        const res = await fetch('/admin/confirm', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({report_id: parseInt(lost), val: 1})
        });
        
        const data = await res.json();
        if (data.ok) {
            showNotification('Match confirmed successfully!', 'success');
            // Optionally refresh the page or update UI
            setTimeout(() => location.reload(), 1500);
        } else {
            showNotification('Failed to confirm match', 'error');
        }
    } catch (error) {
        showNotification('An error occurred', 'error');
        console.error('Error:', error);
    }
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} fade-in`;
    notification.innerHTML = `
        ${message}
        <button class="alert-close" aria-label="Close alert">
            <i class="ri-close-line"></i>
        </button>
    `;
    
    document.querySelector('.main-content').insertAdjacentElement('afterbegin', notification);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease-out forwards';
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

// Shared file upload preview functionality
function initializeFileUploadPreview(inputId, previewId, labelSelector) {
    const fileInput = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const uploadLabel = document.querySelector(labelSelector);
    
    if (!fileInput || !preview) return;
    
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.innerHTML = `
                    <div class="preview-image-container">
                        <img src="${e.target.result}" alt="Preview">
                        <button type="button" class="remove-image" onclick="removeFilePreview('${inputId}', '${previewId}', '${labelSelector}')">
                            <i class="ri-close-circle-line"></i>
                        </button>
                        <div class="file-info">
                            <i class="ri-file-image-line"></i>
                            <span>${file.name}</span>
                        </div>
                    </div>
                `;
                if (uploadLabel) uploadLabel.style.display = 'none';
                preview.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }
    });
}

function removeFilePreview(inputId, previewId, labelSelector) {
    const fileInput = document.getElementById(inputId);
    const preview = document.getElementById(previewId);
    const uploadLabel = document.querySelector(labelSelector);
    
    if (fileInput) fileInput.value = '';
    if (preview) {
        preview.innerHTML = '';
        preview.style.display = 'none';
    }
    if (uploadLabel) uploadLabel.style.display = 'flex';
}
