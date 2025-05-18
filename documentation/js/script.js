// JavaScript for Carbon Credit Verification Documentation

document.addEventListener('DOMContentLoaded', function() {
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
    if (link.getAttribute('href') === currentPath) {
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
  
  // Code syntax highlighting
  document.querySelectorAll('pre code').forEach((block) => {
    if (window.hljs) {
      hljs.highlightBlock(block);
    }
  });
  
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
  
  // Search functionality
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
});
