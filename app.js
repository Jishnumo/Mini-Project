document.addEventListener('DOMContentLoaded', function () {
    var video = document.getElementById('video-bg');

    // Pause the video when it ends
    video.addEventListener('ended', function () {
        video.pause();
    });
});