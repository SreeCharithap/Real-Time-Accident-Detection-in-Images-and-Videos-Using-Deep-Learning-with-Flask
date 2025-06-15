document.addEventListener('DOMContentLoaded', function() {
    // Main content and result views
    const mainContent = document.getElementById('mainContent');
    const imageResultView = document.getElementById('imageResultView');
    const videoResultView = document.getElementById('videoResultView');
    
    // Image elements
    const imageForm = document.getElementById('imageForm');
    const imageAlert = document.getElementById('imageAlert');
    const imageConfidence = document.getElementById('imageConfidence');
    const originalImage = document.getElementById('originalImage');
    const heatmapImage = document.getElementById('heatmapImage');
    const backFromImage = document.getElementById('backFromImage');
    
    // Video elements
    const videoForm = document.getElementById('videoForm');
    const previewVideo = document.getElementById('previewVideo');
    const videoAlert = document.getElementById('videoAlert');
    const videoConfidence = document.getElementById('videoConfidence');
    const randomFrame = document.getElementById('randomFrame');
    const backFromVideo = document.getElementById('backFromVideo');
    const processingMessage = document.getElementById('processingMessage');
    
    // Back button functionality
    backFromImage.addEventListener('click', function() {
        imageResultView.classList.add('d-none');
        mainContent.classList.remove('d-none');
    });
    
    backFromVideo.addEventListener('click', function() {
        videoResultView.classList.add('d-none');
        mainContent.classList.remove('d-none');
    });
    
    // Handle image form submission
    imageForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(imageForm);
        const file = document.getElementById('imageFile').files[0];
        
        if (!file) {
            alert('Please select an image file');
            return;
        }
        
        // Show loading indicator
        const submitBtn = imageForm.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        submitBtn.disabled = true;
        
        // Send the image for classification
        fetch('/classify_image', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset button
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Hide main content and show result view
            mainContent.classList.add('d-none');
            imageResultView.classList.remove('d-none');
            
            // Display result with enhanced styling
            if (data.result === 'Accident') {
                imageAlert.className = 'alert alert-danger';
                imageAlert.innerHTML = '<div class="accident-result">Accident Detected</div>';
            } else {
                imageAlert.className = 'alert alert-success';
                imageAlert.innerHTML = '<div class="non-accident-result">No Accident Detected</div>';
            }
            
            // Display confidence percentage with better formatting
            imageConfidence.innerHTML = '<span class="confidence-value">' + data.confidence.toFixed(2) + '%</span>';
            
            // Display original and heatmap images with labels
            document.getElementById('originalImageContainer').innerHTML = 
                '<div class="image-label">Original Image</div>' +
                '<img src="data:image/jpeg;base64,' + data.original_image + '" class="img-fluid preview-img" id="originalImage">';
                
            document.getElementById('heatmapImageContainer').innerHTML = 
                '<div class="image-label">Heatmap Visualization</div>' +
                '<img src="data:image/jpeg;base64,' + data.heatmap_image + '" class="img-fluid preview-img" id="heatmapImage">';
        })
        .catch(error => {
            // Reset button
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
            
            console.error('Error:', error);
            alert('An error occurred while processing the image');
        });
    });
    
    // Handle video form submission
    videoForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(videoForm);
        const file = document.getElementById('videoFile').files[0];
        
        if (!file) {
            alert('Please select a video file');
            return;
        }
        
        // Show loading indicator
        const submitBtn = videoForm.querySelector('button[type="submit"]');
        const originalBtnText = submitBtn.innerHTML;
        submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        submitBtn.disabled = true;
        
        // Show processing message
        processingMessage.classList.remove('d-none');
        
        // Send the video for classification
        fetch('/classify_video', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            // Reset button and hide processing message
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
            processingMessage.classList.add('d-none');
            
            if (data.error) {
                alert(data.error);
                return;
            }
            
            // Hide main content and show result view
            mainContent.classList.add('d-none');
            videoResultView.classList.remove('d-none');
            
            // Display result with enhanced styling
            if (data.result === 'Accident') {
                videoAlert.className = 'alert alert-danger';
                videoAlert.innerHTML = '<div class="accident-result">Accident Detected</div>';
            } else {
                videoAlert.className = 'alert alert-success';
                videoAlert.innerHTML = '<div class="non-accident-result">No Accident Detected</div>';
            }
            
            // Display confidence percentage with better formatting
            videoConfidence.innerHTML = '<span class="confidence-value">' + data.confidence.toFixed(2) + '%</span>';
            
            // Display video with label
            document.getElementById('videoContainer').innerHTML = 
                '<div class="image-label">Video Preview</div>' +
                '<video controls class="preview-video" id="previewVideo"></video>';
            
            // Display random frame with label
            document.getElementById('randomFrameContainer').innerHTML = 
                '<div class="image-label">Representative Frame</div>' +
                '<img src="data:image/jpeg;base64,' + data.random_frame + '" class="img-fluid preview-img" id="randomFrame">';
                
            // Set video source after adding to DOM
            const videoURL = URL.createObjectURL(file);
            document.getElementById('previewVideo').src = videoURL;
        })
        .catch(error => {
            // Reset button and hide processing message
            submitBtn.innerHTML = originalBtnText;
            submitBtn.disabled = false;
            processingMessage.classList.add('d-none');
            
            console.error('Error:', error);
            alert('An error occurred while processing the video');
        });
    });
});
