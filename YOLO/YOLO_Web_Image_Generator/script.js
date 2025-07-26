document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('image-upload');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Select File.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        if (result.error) {
            alert(result.error);
            return;
        }

        const resultElement = document.getElementById('result');
        const imageElement = document.getElementById('uploaded-image');
        const canvas = document.getElementById('detection-canvas');
        const ctx = canvas.getContext('2d');

        // 이미지 표시
        imageElement.src = result.image_path;
        imageElement.onload = () => {
            resizeImage(imageElement, canvas);

            // 이미지를 Canvas에 그리기
            ctx.drawImage(imageElement, 0, 0, canvas.width, canvas.height);

            // 바운딩 박스 그리기
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.font = '16px Arial';
            ctx.fillStyle = 'red';

            result.detections.forEach(detection => {
                const x = detection.xmin;
                const y = detection.ymin;
                const width = detection.xmax - detection.xmin;
                const height = detection.ymax - detection.ymin;

                ctx.strokeRect(x, y, width, height);
                
                // 클래스 이름과 확률 표시
                const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
                ctx.fillText(label, x, y > 20 ? y - 5 : y + 20);
            });

            // Canvas 크기를 이미지 표시 영역에 맞게 조정
            canvas.style.width = '100%';
            canvas.style.height = 'auto';
        };

        resultElement.classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        alert('오류가 발생했습니다. 다시 시도해주세요.');
    }
});

function resizeImage(image, canvas) {
    const maxWidth = window.innerWidth * 0.8; // 창 너비의 80%
    const maxHeight = window.innerHeight * 0.8; // 창 높이의 80%
    
    let width = image.naturalWidth;
    let height = image.naturalHeight;

    if (width > maxWidth) {
        height = (maxWidth / width) * height;
        width = maxWidth;
    }

    if (height > maxHeight) {
        width = (maxHeight / height) * width;
        height = maxHeight;
    }

    canvas.width = width;
    canvas.height = height;

    image.style.width = width + 'px';
    image.style.height = height + 'px';
}

function enlargeImage() {
    const img = document.getElementById('uploaded-image');
    img.classList.toggle('enlarged');
}

// 이미지 클릭 이벤트 추가
document.getElementById('uploaded-image').addEventListener('click', enlargeImage);

