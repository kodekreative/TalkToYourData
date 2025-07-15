/**
 * Modal Fix Utility
 * Automatically detects and fixes common modal issues:
 * - Scrolling problems
 * - Buttons positioned outside window
 * - Poor responsive behavior
 */

class ModalFixer {
    constructor(options = {}) {
        this.options = {
            modalSelector: '.modal, [role="dialog"], .popup, .wizard',
            overlaySelector: '.modal-overlay, .modal-backdrop, .overlay',
            containerSelector: '.modal-content, .modal-dialog, .modal-container, .popup-content',
            headerSelector: '.modal-header, .popup-header, .wizard-header',
            bodySelector: '.modal-body, .popup-body, .wizard-body',
            footerSelector: '.modal-footer, .popup-footer, .wizard-footer',
            buttonSelector: 'button, .btn, .button',
            autoInit: true,
            maxHeight: '90vh',
            maxWidth: '90vw',
            padding: '20px',
            ...options
        };

        if (this.options.autoInit) {
            this.init();
        }
    }

    init() {
        // Wait for DOM to be ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', () => this.fixAllModals());
        } else {
            this.fixAllModals();
        }

        // Watch for dynamically added modals
        this.observeNewModals();
    }

    fixAllModals() {
        const modals = document.querySelectorAll(this.options.modalSelector);
        modals.forEach(modal => this.fixModal(modal));
    }

    fixModal(modal) {
        if (!modal || modal.hasAttribute('data-modal-fixed')) {
            return;
        }

        console.log('Fixing modal:', modal);

        // Mark as fixed to avoid duplicate processing
        modal.setAttribute('data-modal-fixed', 'true');

        // Fix modal structure
        this.fixModalStructure(modal);
        
        // Fix scrolling
        this.fixScrolling(modal);
        
        // Fix button positioning
        this.fixButtonPositioning(modal);
        
        // Add responsive behavior
        this.makeResponsive(modal);
        
        // Add keyboard navigation
        this.addKeyboardNavigation(modal);
        
        // Fix overlay behavior
        this.fixOverlayBehavior(modal);
    }

    fixModalStructure(modal) {
        // Ensure proper CSS classes and structure
        modal.style.position = 'fixed';
        modal.style.top = '0';
        modal.style.left = '0';
        modal.style.width = '100vw';
        modal.style.height = '100vh';
        modal.style.display = 'flex';
        modal.style.justifyContent = 'center';
        modal.style.alignItems = 'center';
        modal.style.zIndex = '1000';
        modal.style.backgroundColor = 'rgba(0, 0, 0, 0.5)';
        modal.style.padding = this.options.padding;
        modal.style.boxSizing = 'border-box';

        // Find or create container
        let container = modal.querySelector(this.options.containerSelector);
        if (!container) {
            // Wrap existing content in a container
            const existingContent = Array.from(modal.children);
            container = document.createElement('div');
            container.className = 'modal-container-fixed';
            
            existingContent.forEach(child => {
                if (!child.classList.contains('modal-overlay') && 
                    !child.classList.contains('modal-backdrop')) {
                    container.appendChild(child);
                }
            });
            
            modal.appendChild(container);
        }

        // Style the container
        container.style.background = 'white';
        container.style.borderRadius = '8px';
        container.style.boxShadow = '0 10px 30px rgba(0, 0, 0, 0.3)';
        container.style.maxWidth = this.options.maxWidth;
        container.style.maxHeight = this.options.maxHeight;
        container.style.width = '800px';
        container.style.display = 'flex';
        container.style.flexDirection = 'column';
        container.style.position = 'relative';
        container.style.margin = 'auto';
        container.style.overflow = 'hidden';
    }

    fixScrolling(modal) {
        const container = modal.querySelector(this.options.containerSelector) || 
                         modal.querySelector('.modal-container-fixed');
        
        if (!container) return;

        // Find header, body, and footer
        const header = container.querySelector(this.options.headerSelector);
        const footer = container.querySelector(this.options.footerSelector);
        let body = container.querySelector(this.options.bodySelector);

        // If no body found, create one from remaining content
        if (!body) {
            const allChildren = Array.from(container.children);
            const nonBodyElements = allChildren.filter(child => 
                child === header || child === footer || 
                child.classList.contains('modal-close') ||
                child.classList.contains('close-button')
            );
            
            if (allChildren.length > nonBodyElements.length) {
                body = document.createElement('div');
                body.className = 'modal-body-fixed';
                
                allChildren.forEach(child => {
                    if (!nonBodyElements.includes(child)) {
                        body.appendChild(child);
                    }
                });
                
                // Insert body before footer or at end
                if (footer) {
                    container.insertBefore(body, footer);
                } else {
                    container.appendChild(body);
                }
            }
        }

        // Style header
        if (header) {
            header.style.padding = '20px 24px';
            header.style.borderBottom = '1px solid #e5e7eb';
            header.style.backgroundColor = '#f9fafb';
            header.style.borderRadius = '8px 8px 0 0';
            header.style.flexShrink = '0';
        }

        // Style body for scrolling
        if (body) {
            body.style.padding = '24px';
            body.style.overflowY = 'auto';
            body.style.overflowX = 'hidden';
            body.style.flex = '1';
            body.style.minHeight = '0'; // Important for flexbox scrolling
        }

        // Style footer
        if (footer) {
            footer.style.padding = '16px 24px';
            footer.style.borderTop = '1px solid #e5e7eb';
            footer.style.backgroundColor = '#f9fafb';
            footer.style.borderRadius = '0 0 8px 8px';
            footer.style.display = 'flex';
            footer.style.justifyContent = 'flex-end';
            footer.style.gap = '12px';
            footer.style.flexShrink = '0';
            footer.style.position = 'sticky';
            footer.style.bottom = '0';
        }
    }

    fixButtonPositioning(modal) {
        const footer = modal.querySelector(this.options.footerSelector);
        if (!footer) return;

        const buttons = footer.querySelectorAll(this.options.buttonSelector);
        
        buttons.forEach(button => {
            // Ensure buttons are properly styled
            button.style.padding = '8px 16px';
            button.style.borderRadius = '6px';
            button.style.border = '1px solid #d1d5db';
            button.style.backgroundColor = 'white';
            button.style.color = '#374151';
            button.style.cursor = 'pointer';
            button.style.fontSize = '14px';
            button.style.fontWeight = '500';
            button.style.transition = 'all 0.2s ease';

            // Add hover effects
            button.addEventListener('mouseenter', () => {
                if (!button.classList.contains('primary')) {
                    button.style.backgroundColor = '#f3f4f6';
                    button.style.borderColor = '#9ca3af';
                }
            });

            button.addEventListener('mouseleave', () => {
                if (!button.classList.contains('primary')) {
                    button.style.backgroundColor = 'white';
                    button.style.borderColor = '#d1d5db';
                }
            });

            // Style primary buttons
            if (button.classList.contains('primary') || 
                button.type === 'submit' ||
                button.textContent.toLowerCase().includes('save') ||
                button.textContent.toLowerCase().includes('next') ||
                button.textContent.toLowerCase().includes('submit')) {
                button.style.backgroundColor = '#3b82f6';
                button.style.borderColor = '#3b82f6';
                button.style.color = 'white';
            }
        });
    }

    makeResponsive(modal) {
        const applyResponsiveStyles = () => {
            const container = modal.querySelector(this.options.containerSelector) || 
                             modal.querySelector('.modal-container-fixed');
            
            if (!container) return;

            if (window.innerWidth <= 768) {
                modal.style.padding = '10px';
                container.style.maxWidth = '95vw';
                container.style.maxHeight = '95vh';
                container.style.width = '100%';

                // Adjust padding for mobile
                const header = container.querySelector(this.options.headerSelector);
                const body = container.querySelector(this.options.bodySelector) || 
                           container.querySelector('.modal-body-fixed');
                const footer = container.querySelector(this.options.footerSelector);

                [header, body, footer].forEach(element => {
                    if (element) {
                        element.style.padding = '16px';
                    }
                });
            } else {
                modal.style.padding = this.options.padding;
                container.style.maxWidth = this.options.maxWidth;
                container.style.maxHeight = this.options.maxHeight;
                container.style.width = '800px';
            }
        };

        applyResponsiveStyles();
        window.addEventListener('resize', applyResponsiveStyles);
    }

    addKeyboardNavigation(modal) {
        // Close on Escape
        const handleKeydown = (e) => {
            if (e.key === 'Escape') {
                this.closeModal(modal);
            }
        };

        document.addEventListener('keydown', handleKeydown);

        // Store handler for cleanup
        modal._keydownHandler = handleKeydown;
    }

    fixOverlayBehavior(modal) {
        const container = modal.querySelector(this.options.containerSelector) || 
                         modal.querySelector('.modal-container-fixed');
        
        if (!container) return;

        // Prevent modal close when clicking inside container
        container.addEventListener('click', (e) => {
            e.stopPropagation();
        });

        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal(modal);
            }
        });

        // Prevent body scroll when modal is open
        document.body.style.overflow = 'hidden';
    }

    closeModal(modal) {
        // Remove keyboard handler
        if (modal._keydownHandler) {
            document.removeEventListener('keydown', modal._keydownHandler);
        }

        // Restore body scroll
        document.body.style.overflow = '';

        // Hide modal
        modal.style.display = 'none';
        
        // Trigger close event
        modal.dispatchEvent(new CustomEvent('modalClosed', { bubbles: true }));
    }

    observeNewModals() {
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                mutation.addedNodes.forEach((node) => {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        // Check if the added node is a modal
                        if (node.matches && node.matches(this.options.modalSelector)) {
                            this.fixModal(node);
                        }
                        
                        // Check if the added node contains modals
                        const modals = node.querySelectorAll && node.querySelectorAll(this.options.modalSelector);
                        if (modals) {
                            modals.forEach(modal => this.fixModal(modal));
                        }
                    }
                });
            });
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });
    }
}

// Auto-initialize when script loads
const modalFixer = new ModalFixer();

// Export for manual use
window.ModalFixer = ModalFixer;
window.modalFixer = modalFixer;

// Provide manual fix function
window.fixModal = function(selector) {
    const modal = typeof selector === 'string' ? document.querySelector(selector) : selector;
    if (modal) {
        modalFixer.fixModal(modal);
    }
};

console.log('Modal Fixer loaded and initialized. Use window.fixModal(selector) to manually fix specific modals.'); 