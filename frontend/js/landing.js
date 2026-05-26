// Landing page animations and interactions
document.addEventListener('DOMContentLoaded', () => {
  // Animate steps on scroll
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry, i) => {
      if (entry.isIntersecting) {
        setTimeout(() => {
          entry.target.style.opacity = '1';
          entry.target.style.transform = 'translateY(0)';
        }, i * 100);
      }
    });
  }, { threshold: 0.1 });

  document.querySelectorAll('.step-card').forEach(card => {
    card.style.opacity = '0';
    card.style.transform = 'translateY(30px)';
    card.style.transition = 'all 0.5s ease';
    observer.observe(card);
  });

  // Animate feature pills
  document.querySelectorAll('.feature-pill').forEach((pill, i) => {
    pill.style.opacity = '0';
    pill.style.transform = 'scale(0.9)';
    pill.style.transition = `all 0.3s ease ${i * 0.05}s`;
    setTimeout(() => {
      pill.style.opacity = '1';
      pill.style.transform = 'scale(1)';
    }, 800 + i * 50);
  });
});