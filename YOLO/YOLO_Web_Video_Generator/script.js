document.getElementById('upload-form').addEventListener('submit', async (e) => {
    e.preventDefault();

    const fileInput = document.getElementById('video-upload');
    const file = fileInput.files[0];

    if (!file) {
        alert('파일을 선택해주세요.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('/detect', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();

        if (result.error) {
            alert(result.error);
            return;
        }

        console.log(result);

        const resultElement = document.getElementById('result');
        const videoElement = document.getElementById('uploaded-video');

        // 동영상 표시
        videoElement.src = result.video_path;
        videoElement.load();
        videoElement.style.display = 'block'; // Ensure the video element is visible
        videoElement.controls = true; // Ensure controls are enabled

        resultElement.classList.remove('hidden');
    } catch (error) {
        console.error('Error:', error);
        alert('오류가 발생했습니다. 다시 시도해주세요.');
    }
});
