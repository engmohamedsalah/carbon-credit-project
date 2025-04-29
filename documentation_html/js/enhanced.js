// Enhanced JavaScript for user-friendly documentation

document.addEventListener('DOMContentLoaded', function() {
  // Original script functionality
  initializeNavigation();
  initializeSearch();
  initializeCodeHighlighting();
  initializeSmoothScrolling();
  initializeTableOfContents();
  initializeImageLightbox();
  
  // New enhanced functionality
  initializeDarkMode();
  initializeGlossary();
  initializeProgressTracker();
  initializeInteractiveDiagrams();
  initializeFaqAccordion();
  initializePrintButton();
  initializeReadingTimeEstimate();
  initializeSimpleExplanationToggles();
});

// Original functionality
function initializeNavigation() {
  // Mobile navigation toggle
  const navToggle = document.getElementById('nav-toggle');
  const sidebar = document.querySelector('.sidebar');
  
  if (navToggle) {
    navToggle.addEventListener('click', function() {
      sidebar.classList.toggle('active');
    });
  }
  
  // Highlight active navigation item
  const currentPath = window.location.pathname;
  const navLinks = document.querySelectorAll('.nav-link');
  
  navLinks.forEach(link => {
    if (link.getAttribute('href') === currentPath.split('/').pop()) {
      link.classList.add('active');
      
      // Expand parent if in subnav
      const parentLi = link.closest('li.nav-item');
      if (parentLi && parentLi.parentElement.classList.contains('subnav')) {
        const parentNav = parentLi.parentElement.previousElementSibling;
        if (parentNav && parentNav.classList.contains('nav-link-toggle')) {
          parentNav.classList.add('active');
          parentLi.parentElement.style.display = 'block';
        }
      }
    }
  });
  
  // Collapsible sections
  const toggleButtons = document.querySelectorAll('.nav-link-toggle');
  
  toggleButtons.forEach(button => {
    button.addEventListener('click', function(e) {
      e.preventDefault();
      const subnav = this.nextElementSibling;
      
      if (subnav.style.display === 'block') {
        subnav.style.display = 'none';
        this.classList.remove('active');
      } else {
        subnav.style.display = 'block';
        this.classList.add('active');
      }
    });
  });
}

function initializeSearch() {
  const searchInput = document.getElementById('search-input');
  const searchResults = document.getElementById('search-results');
  
  if (searchInput && searchResults) {
    searchInput.addEventListener('input', function() {
      const query = this.value.toLowerCase().trim();
      
      if (query.length < 2) {
        searchResults.innerHTML = '';
        searchResults.style.display = 'none';
        return;
      }
      
      // Simple search through content
      const contentElements = document.querySelectorAll('.content-section h2, .content-section h3, .content-section p');
      const matches = [];
      
      contentElements.forEach(element => {
        const text = element.textContent.toLowerCase();
        if (text.includes(query)) {
          let heading = element;
          let headingText = element.textContent;
          
          // If it's a paragraph, find the nearest heading
          if (element.tagName === 'P') {
            let currentElement = element.previousElementSibling;
            while (currentElement) {
              if (currentElement.tagName === 'H2' || currentElement.tagName === 'H3') {
                heading = currentElement;
                headingText = currentElement.textContent;
                break;
              }
              currentElement = currentElement.previousElementSibling;
            }
          }
          
          // Add to matches if not already included
          if (!matches.some(match => match.element === heading)) {
            matches.push({
              element: heading,
              text: headingText,
              id: heading.id || ''
            });
          }
        }
      });
      
      // Display results
      if (matches.length > 0) {
        searchResults.innerHTML = '';
        const resultsList = document.createElement('ul');
        
        matches.forEach(match => {
          const listItem = document.createElement('li');
          const link = document.createElement('a');
          link.textContent = match.text;
          
          if (match.id) {
            link.href = '#' + match.id;
            link.addEventListener('click', function() {
              searchResults.style.display = 'none';
            });
          }
          
          listItem.appendChild(link);
          resultsList.appendChild(listItem);
        });
        
        searchResults.appendChild(resultsList);
        searchResults.style.display = 'block';
      } else {
        searchResults.innerHTML = '<p>No results found</p>';
        searchResults.style.display = 'block';
      }
    });
    
    // Close search results when clicking outside
    document.addEventListener('click', function(e) {
      if (e.target !== searchInput && e.target !== searchResults) {
        searchResults.style.display = 'none';
      }
    });
  }
}

function initializeCodeHighlighting() {
  // Code syntax highlighting
  document.querySelectorAll('pre code').forEach((block) => {
    if (window.hljs) {
      hljs.highlightBlock(block);
    }
  });
}

function initializeSmoothScrolling() {
  // Smooth scrolling for anchor links
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
      e.preventDefault();
      
      const targetId = this.getAttribute('href');
      const targetElement = document.querySelector(targetId);
      
      if (targetElement) {
        window.scrollTo({
          top: targetElement.offsetTop - 20,
          behavior: 'smooth'
        });
      }
    });
  });
}

function initializeTableOfContents() {
  // Table of contents generation
  const tocContainer = document.getElementById('toc');
  if (tocContainer) {
    const headings = document.querySelectorAll('.content-section h2, .content-section h3');
    if (headings.length > 0) {
      const tocList = document.createElement('ul');
      tocList.classList.add('toc-list');
      
      headings.forEach((heading, index) => {
        // Add ID to heading if it doesn't have one
        if (!heading.id) {
          heading.id = 'heading-' + index;
        }
        
        const listItem = document.createElement('li');
        listItem.classList.add(heading.tagName.toLowerCase() + '-item');
        
        const link = document.createElement('a');
        link.textContent = heading.textContent;
        link.href = '#' + heading.id;
        
        listItem.appendChild(link);
        tocList.appendChild(listItem);
      });
      
      tocContainer.appendChild(tocList);
    }
  }
}

function initializeImageLightbox() {
  // Image lightbox
  const contentImages = document.querySelectorAll('.content-section img:not(.no-lightbox)');
  
  contentImages.forEach(img => {
    img.addEventListener('click', function() {
      const lightbox = document.createElement('div');
      lightbox.classList.add('lightbox');
      
      const lightboxImg = document.createElement('img');
      lightboxImg.src = this.src;
      
      const closeBtn = document.createElement('span');
      closeBtn.classList.add('lightbox-close');
      closeBtn.innerHTML = '&times;';
      closeBtn.addEventListener('click', function() {
        document.body.removeChild(lightbox);
      });
      
      lightbox.appendChild(closeBtn);
      lightbox.appendChild(lightboxImg);
      
      document.body.appendChild(lightbox);
      
      lightbox.addEventListener('click', function(e) {
        if (e.target === lightbox) {
          document.body.removeChild(lightbox);
        }
      });
    });
    
    // Add cursor pointer to indicate clickable
    img.style.cursor = 'pointer';
  });
}

// New enhanced functionality
function initializeDarkMode() {
  // Create dark mode toggle button if it doesn't exist
  if (!document.querySelector('.dark-mode-toggle')) {
    const darkModeToggle = document.createElement('div');
    darkModeToggle.classList.add('dark-mode-toggle');
    darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
    document.body.appendChild(darkModeToggle);
    
    // Check for saved preference
    const darkModePreference = localStorage.getItem('darkMode');
    if (darkModePreference === 'enabled') {
      document.body.classList.add('dark-mode');
      darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    // Toggle dark mode
    darkModeToggle.addEventListener('click', function() {
      document.body.classList.toggle('dark-mode');
      
      if (document.body.classList.contains('dark-mode')) {
        localStorage.setItem('darkMode', 'enabled');
        darkModeToggle.innerHTML = '<i class="fas fa-sun"></i>';
      } else {
        localStorage.setItem('darkMode', 'disabled');
        darkModeToggle.innerHTML = '<i class="fas fa-moon"></i>';
      }
    });
  }
}

function initializeGlossary() {
  // Find all glossary terms
  const glossaryTerms = document.querySelectorAll('.glossary-term');
  
  glossaryTerms.forEach(term => {
    // Create popup for each term
    const popup = document.createElement('div');
    popup.classList.add('glossary-popup');
    popup.textContent = term.getAttribute('data-definition');
    document.body.appendChild(popup);
    
    // Show popup on hover
    term.addEventListener('mouseenter', function(e) {
      const rect = term.getBoundingClientRect();
      popup.style.left = rect.left + 'px';
      popup.style.top = (rect.bottom + 10) + 'px';
      popup.style.display = 'block';
    });
    
    // Hide popup when mouse leaves
    term.addEventListener('mouseleave', function() {
      popup.style.display = 'none';
    });
  });
}

function initializeProgressTracker() {
  // Create progress tracker if it doesn't exist
  if (!document.querySelector('.progress-tracker')) {
    const progressTracker = document.createElement('div');
    progressTracker.classList.add('progress-tracker');
    document.body.appendChild(progressTracker);
    
    // Update progress on scroll
    window.addEventListener('scroll', function() {
      const windowHeight = window.innerHeight;
      const documentHeight = document.documentElement.scrollHeight;
      const scrollTop = window.scrollY || document.documentElement.scrollTop;
      
      const scrollPercent = (scrollTop / (documentHeight - windowHeight)) * 100;
      progressTracker.style.width = scrollPercent + '%';
    });
  }
}

function initializeInteractiveDiagrams() {
  // Find all interactive diagrams
  const diagrams = document.querySelectorAll('.interactive-diagram');
  
  diagrams.forEach(diagram => {
    // Find all hotspots in this diagram
    const hotspots = diagram.querySelectorAll('.diagram-hotspot');
    
    hotspots.forEach(hotspot => {
      const tooltip = document.createElement('div');
      tooltip.classList.add('diagram-tooltip');
      tooltip.innerHTML = hotspot.getAttribute('data-tooltip');
      hotspot.appendChild(tooltip);
      
      // Show tooltip on click
      hotspot.addEventListener('click', function(e) {
        e.stopPropagation();
        
        // Hide all other tooltips
        document.querySelectorAll('.diagram-tooltip').forEach(t => {
          if (t !== tooltip) {
            t.style.display = 'none';
          }
        });
        
        // Toggle this tooltip
        if (tooltip.style.display === 'block') {
          tooltip.style.display = 'none';
        } else {
          tooltip.style.display = 'block';
        }
      });
    });
    
    // Hide tooltips when clicking elsewhere
    document.addEventListener('click', function() {
      document.querySelectorAll('.diagram-tooltip').forEach(tooltip => {
        tooltip.style.display = 'none';
      });
    });
  });
}

function initializeFaqAccordion() {
  // Find all FAQ items
  const faqQuestions = document.querySelectorAll('.faq-question');
  
  faqQuestions.forEach(question => {
    question.addEventListener('click', function() {
      // Toggle active class
      this.classList.toggle('active');
      
      // Toggle answer visibility
      const answer = this.nextElementSibling;
      answer.classList.toggle('active');
    });
  });
}

function initializePrintButton() {
  // Create print button if it doesn't exist
  if (!document.querySelector('.print-button') && document.querySelector('.content-header')) {
    const printButton = document.createElement('button');
    printButton.classList.add('print-button');
    printButton.innerHTML = '<i class="fas fa-print"></i> Print';
    document.querySelector('.content-header').appendChild(printButton);
    
    // Print page when clicked
    printButton.addEventListener('click', function() {
      window.print();
    });
  }
}

function initializeReadingTimeEstimate() {
  // Create reading time element if it doesn't exist
  if (!document.querySelector('.reading-time') && document.querySelector('.content-header')) {
    const readingTime = document.createElement('div');
    readingTime.classList.add('reading-time');
    
    // Calculate reading time (average reading speed: 200 words per minute)
    const content = document.querySelector('.content-section').textContent;
    const wordCount = content.split(/\s+/).length;
    const readingTimeMinutes = Math.ceil(wordCount / 200);
    
    readingTime.innerHTML = `<i class="fas fa-clock"></i> ${readingTimeMinutes} min read`;
    document.querySelector('.content-header').appendChild(readingTime);
  }
}

function initializeSimpleExplanationToggles() {
  // Add toggle buttons to all simple explanations
  const simpleExplanations = document.querySelectorAll('.simple-explanation');
  
  simpleExplanations.forEach(explanation => {
    // Add toggle button if it doesn't exist
    if (!explanation.querySelector('.toggle-technical')) {
      const toggleButton = document.createElement('button');
      toggleButton.classList.add('toggle-technical');
      toggleButton.textContent = 'Show Technical Details';
      explanation.appendChild(toggleButton);
      
      // Create technical explanation div if it doesn't exist
      if (!explanation.nextElementSibling || !explanation.nextElementSibling.classList.contains('technical-explanation')) {
        const technicalExplanation = document.createElement('div');
        technicalExplanation.classList.add('technical-explanation');
        technicalExplanation.style.display = 'none';
        
        // Move technical content here if specified
        const technicalContentId = explanation.getAttribute('data-technical-content');
        if (technicalContentId) {
          const technicalContent = document.getElementById(technicalContentId);
          if (technicalContent) {
            technicalExplanation.innerHTML = technicalContent.innerHTML;
            technicalContent.style.display = 'none';
          }
        }
        
        explanation.parentNode.insertBefore(technicalExplanation, explanation.nextSibling);
        
        // Toggle technical explanation
        toggleButton.addEventListener('click', function() {
          const technical = explanation.nextElementSibling;
          if (technical.style.display === 'none') {
            technical.style.display = 'block';
            toggleButton.textContent = 'Hide Technical Details';
          } else {
            technical.style.display = 'none';
            toggleButton.textContent = 'Show Technical Details';
          }
        });
      }
    }
  });
}
