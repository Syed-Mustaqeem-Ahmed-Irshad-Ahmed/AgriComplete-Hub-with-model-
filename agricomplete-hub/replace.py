import sys

new_func = """function analyzeDisease() {
  const result = document.getElementById('diagnosisResult');
  const placeholder = document.getElementById('diagnosisPlaceholder');
  const info = document.getElementById('diseaseInfo');
  const badge = document.getElementById('resultBadge');
  const fileInput = document.getElementById('leafInput');

  if (!fileInput.files || fileInput.files.length === 0) {
    alert('Please upload an image first.');
    return;
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append('image', file);

  // Show loading state
  if (placeholder) placeholder.style.display = 'none';
  if (result) result.style.display = 'block';
  if (info) info.innerHTML = '<div style=\"text-align:center;padding:20px;\"><i class=\"fas fa-spinner fa-spin\" style=\"font-size:2rem;color:var(--color-primary);\"></i><p>Analyzing image...</p></div>';

  fetch('http://localhost:5000/api/disease/predict', {
    method: 'POST',
    body: formData
  })
  .then(response => {
    if (!response.ok) {
      throw new Error('Analysis failed');
    }
    return response.json();
  })
  .then(disease => {
    if (badge) {
      badge.className = `badge ${disease.badgeClass}`;
      badge.innerHTML = `<i class=\"fas fa-exclamation-triangle\"></i> ${disease.severity} Severity`;
    }
    
    if (info) {
      info.innerHTML = `
        <h3 style=\"margin-bottom:var(--space-sm);color:var(--text-primary);\">${disease.name}</h3>
        <div style=\"margin-bottom:var(--space-md);\">
          <div style=\"display:flex;justify-content:space-between;font-size:.85rem;margin-bottom:4px;\">
            <span>Confidence Score</span>
            <strong style=\"color:var(--color-primary);\">${disease.confidence}%</strong>
          </div>
          <div class=\"confidence-bar\">
            <div class=\"confidence-fill\" style=\"width:${disease.confidence}%;\"></div>
          </div>
        </div>
        <p style=\"font-size:.88rem;margin-bottom:var(--space-md);\">${disease.description}</p>
        
        <h4 style=\"font-size:.9rem;margin-bottom:var(--space-sm);\"><i class=\"fas fa-exclamation-circle\" style=\"color:#C62828;margin-right:6px;\"></i>Symptoms</h4>
        <ul style=\"font-size:.85rem;color:var(--text-secondary);margin-bottom:var(--space-md);padding-left:16px;\">
          ${disease.symptoms.map(s => `<li style=\"margin-bottom:4px;list-style:disc;\">${s}</li>`).join('')}
        </ul>
        
        <h4 style=\"font-size:.9rem;margin-bottom:var(--space-sm);\"><i class=\"fas fa-prescription-bottle-alt\" style=\"color:var(--color-primary);margin-right:6px;\"></i>Treatment</h4>
        <ul style=\"font-size:.85rem;color:var(--text-secondary);margin-bottom:var(--space-md);padding-left:16px;\">
          ${disease.treatment.map(t => `<li style=\"margin-bottom:4px;list-style:disc;\">${t}</li>`).join('')}
        </ul>
        
        <h4 style=\"font-size:.9rem;margin-bottom:var(--space-sm);\"><i class=\"fas fa-shield-alt\" style=\"color:var(--color-accent);margin-right:6px;\"></i>Prevention</h4>
        <ul style=\"font-size:.85rem;color:var(--text-secondary);padding-left:16px;\">
          ${disease.prevention.map(p => `<li style=\"margin-bottom:4px;list-style:disc;\">${p}</li>`).join('')}
        </ul>
      `;
    }
  })
  .catch(error => {
    console.error(error);
    if (info) info.innerHTML = '<div style=\"text-align:center;padding:20px;color:red;\"><i class=\"fas fa-exclamation-circle\"></i><p>Error analyzing image. Server might be down or image is invalid.</p></div>';
  });
}
"""

with open('frontend/js/main.js', 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

start_idx = 1369
end_idx = 1448
new_lines = lines[:start_idx] + [new_func] + lines[end_idx:]

with open('frontend/js/main.js', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)

print("Replacement complete.")
