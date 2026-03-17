const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const mnistPreview = document.getElementById('mnistPreview');
let drawing = false;

function resetBlackBackground() {
    ctx.fillStyle = 'black';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
}

resetBlackBackground();

canvas.addEventListener('mousedown', () => drawing = true);
canvas.addEventListener('mouseup', () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);

function draw(e) {
    if (!drawing) return;
    const rect = canvas.getBoundingClientRect();
    ctx.lineWidth = 15;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.lineTo(e.clientX - rect.left, e.clientY - rect.top);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - rect.left, e.clientY - rect.top);
}

function clearCanvas() {
    resetBlackBackground();
    document.getElementById('result').textContent = 'You wrote: ';
    mnistPreview.removeAttribute('src');
}
async function predict() {
    const imageData = canvas.toDataURL('image/png');

    const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: imageData })
    });

    const data = await res.json();
    document.getElementById('result').textContent =
        `You wrote: ${data.digit} - ${data.confidence}% confident`;

    if (data.mnist_image) {
        mnistPreview.src = data.mnist_image;
    } else {
        mnistPreview.removeAttribute('src');
    }
}