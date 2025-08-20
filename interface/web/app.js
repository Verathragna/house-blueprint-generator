const form = document.getElementById('generate-form');
const logsList = document.getElementById('logs');
const progressSection = document.getElementById('progress');
const resultSection = document.getElementById('result');
const svgContainer = document.getElementById('svg-container');
const downloadJson = document.getElementById('download-json');
const downloadSvg = document.getElementById('download-svg');

function addLog(message) {
  const li = document.createElement('li');
  li.textContent = message;
  logsList.appendChild(li);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  logsList.innerHTML = '';
  progressSection.style.display = 'block';
  resultSection.style.display = 'none';

  const width = parseFloat(document.getElementById('width').value);
  const depth = parseFloat(document.getElementById('depth').value);
  const bedrooms = parseInt(document.getElementById('bedrooms').value, 10);
  const fullBath = parseInt(document.getElementById('bath_full').value, 10);
  const halfBath = parseInt(document.getElementById('bath_half').value, 10);
  const adjacencyText = document.getElementById('adjacency').value.trim();

  let roomAdjacency;
  if (adjacencyText) {
    try {
      roomAdjacency = JSON.parse(adjacencyText);
    } catch (err) {
      alert('Invalid JSON for room adjacency');
      return;
    }
  }

  const params = {
    dimensions: { width, depth },
    bedrooms,
    bathrooms: { full: fullBath, half: halfBath },
  };
  if (roomAdjacency) {
    params.roomAdjacency = roomAdjacency;
  }

  addLog('Submitting job...');

  const response = await fetch('/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ params })
  });
  const data = await response.json();
  const jobId = data.job_id;
  addLog('Job queued with ID ' + jobId);

  const protocol = (location.protocol === 'https:') ? 'wss' : 'ws';
  const ws = new WebSocket(`${protocol}://${location.host}/ws/${jobId}`);

  ws.onmessage = (event) => {
    const msg = JSON.parse(event.data);
    if (msg.log) {
      addLog(msg.log);
    } else if (msg.status === 'completed') {
      ws.close();
      addLog('Completed in ' + msg.result.metadata.processing_time.toFixed(2) + 's');

      svgContainer.innerHTML = `<img src="${msg.result.svg_data_url}" alt="Blueprint" />`;

      const jsonBlob = new Blob([JSON.stringify(msg.result.layout, null, 2)], { type: 'application/json' });
      const jsonUrl = URL.createObjectURL(jsonBlob);
      downloadJson.href = jsonUrl;
      downloadJson.download = msg.result.json_filename || 'layout.json';

      const svgData = atob(msg.result.svg_data_url.split(',')[1]);
      const svgBlob = new Blob([svgData], { type: 'image/svg+xml' });
      const svgUrl = URL.createObjectURL(svgBlob);
      downloadSvg.href = svgUrl;
      downloadSvg.download = msg.result.svg_filename || 'layout.svg';

      resultSection.style.display = 'block';
    } else if (msg.status === 'failed') {
      ws.close();
      addLog('Error: ' + msg.error);
    }
  };
});
